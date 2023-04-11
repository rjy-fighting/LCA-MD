import _init_paths
from numbers import Number
import os
import sys
import datetime
import pprint
import sys

sys.path.append('.')
from lib.core.dataset import build_dataset
from config.default import cfg_from_list, cfg_from_file, update_config
from config.default import config as cfg
from core.engine import creat_data_loader, \
    AverageMeter, accuracy, list2acc, adjust_learning_rate_normal
from core.functions import prepare_env
from utils import mkdir, Logger
from cams_deit import evaluate_cls_loc
from test_cam import val_loc_one_epoch
from lib.core.loss import *
import json
import torch
from torch.utils.tensorboard import SummaryWriter
from lib.optimizer import PolyOptimizer
from timm.models import create_model as create_deit_model
from timm.optim import create_optimizer
import numpy as np
from re import compile

CUBV2 = False


def creat_model(cfg, args):
    print('==> Preparing networks for baseline...')
    # use gpu
    device = torch.device("cuda")
    assert torch.cuda.is_available(), "CUDA is not available"
    # model and optimizer
    model = create_deit_model(
        cfg.MODEL.ARCH,
        pretrained=True,
        num_classes=cfg.DATA.NUM_CLASSES,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )
    if args.resume:
        checkpoint = torch.load(args.resume)
        pretrained_dict = {}

        for k, v in checkpoint['state_dict'].items():
            k_ = '.'.join(k.split('.')[1:])

            pretrained_dict.update({k_: v})

        model.load_state_dict(pretrained_dict)
        print('load pretrained ts-cam model.')
    optimizer = create_optimizer(args, model)
    model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count()))).to(device)
    # loss
    cls_criterion = torch.nn.CrossEntropyLoss().to(device)
    print('Preparing networks done!')
    return device, model, optimizer, cls_criterion


def main():
    args = update_config()
    # create checkpoint directory
    cfg.BASIC.SAVE_DIR = os.path.join(cfg.BASIC.SAVE_ROOT, 'ckpt', cfg.DATA.DATASET,
                                      '{}_CAM-NORMAL_SEED{}_CAM-THR{}_BS{}_{}'.format(
                                          cfg.MODEL.ARCH, cfg.BASIC.SEED, cfg.MODEL.CAM_THR, cfg.TRAIN.BATCH_SIZE,
                                          cfg.BASIC.TIME))
    cfg.BASIC.ROOT_DIR = os.path.join(os.path.dirname(__file__), '..')
    log_dir = os.path.join(cfg.BASIC.SAVE_DIR, 'log');
    mkdir(log_dir)
    ckpt_dir = os.path.join(cfg.BASIC.SAVE_DIR, 'ckpt');
    mkdir(ckpt_dir)
    log_file = os.path.join(cfg.BASIC.SAVE_DIR, 'Log_' + cfg.BASIC.TIME + '.txt')
    # prepare running environment for the whole project
    prepare_env(cfg)

    # start loging
    sys.stdout = Logger(log_file)
    pprint.pprint(cfg)
    writer = SummaryWriter(log_dir)

    train_loader, test_loader, val_loader = creat_data_loader(cfg, os.path.join(cfg.BASIC.ROOT_DIR, cfg.DATA.DATADIR))
    device, model, optimizer, cls_criterion = creat_model(cfg, args)

    max_step = len(train_loader) // cfg.TRAIN.BATCH_SIZE * cfg.SOLVER.NUM_EPOCHS
    criterion2 = [SimMaxLoss(metric='cos', alpha=0.75).cuda(), SimMinLoss(metric='cos').cuda(),
                  SimMaxLoss(metric='cos', alpha=0.75).cuda()]
    best_gtknown = 0
    best_top1_loc = 0
    update_train_step = 0
    update_val_step = 0
    opt_thred = -1
    for epoch in range(1, cfg.SOLVER.NUM_EPOCHS + 1):
        adjust_learning_rate_normal(optimizer, epoch, cfg)
        update_train_step, loss_train, cls_top1_train, cls_top5_train = \
            train_one_epoch(train_loader, model, device, cls_criterion, criterion2,
                            optimizer, epoch, writer, cfg, update_train_step)
        if CUBV2:
            eval_results = val_loc_one_epoch(val_loader, model, device, )
        else:
            eval_results = val_loc_one_epoch(test_loader, model, device, )
        eval_results['epoch'] = epoch
        with open(os.path.join(cfg.BASIC.SAVE_DIR, 'val.txt'), 'a') as val_file:
            val_file.write(json.dumps(eval_results))
            val_file.write('\n')

        loc_gt_known = eval_results['GT-Known_top-1']
        thred = eval_results['det_optThred_thr_50.00_top-1']
        if loc_gt_known >= best_gtknown:
            best_gtknown = loc_gt_known
            torch.save({
                "epoch": epoch,
                'state_dict': model.state_dict(),
                'best_map': best_gtknown
            }, os.path.join(ckpt_dir, f'model_best.pth'))
            opt_thred = thred
        print("Best GT_LOC: {}".format(best_gtknown))
        print("Best TOP1_LOC: {}".format(best_gtknown))
        print(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))
    if CUBV2:
        print('Testing...')
        checkpoint = torch.load(os.path.join(ckpt_dir, f'model_best.pth'))
        pretrained_dict = {}

        for k, v in checkpoint['state_dict'].items():
            k_ = '.'.join(k.split('.')[1:])
            pretrained_dict.update({k_: v})

        model.load_state_dict(pretrained_dict)
        eval_results = val_loc_one_epoch(test_loader, model, device, opt_thred=opt_thred)
        for k, v in eval_results.items():
            if isinstance(v, np.ndarray):
                v = [round(out, 2) for out in v.tolist()]
            elif isinstance(v, Number):
                v = round(v, 2)
            else:
                raise ValueError(f'Unsupport metric type: {type(v)}')
            print(f'\n{k} : {v}')
        with open(os.path.join(cfg.BASIC.SAVE_DIR, 'test.txt'), 'a') as test_file:
            test_file.write(json.dumps(eval_results))
            test_file.write('\n')


def train_one_epoch(train_loader, model, device, criterion, criterion2, optimizer, epoch,
                    writer, cfg, update_train_step):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    log_var = ['module.layers.[0-9]+.fuse._loss_rate', 'module.layers.[0-9]+.thred']
    log_scopes = [compile(log_scope) for log_scope in log_var]
    δ = 0.6
    model.train()
    for i, (input, target) in enumerate(train_loader):
        # update iteration steps
        update_train_step += 1

        target = target.to(device)
        input = input.to(device)
        vars = {}
        for log_scope in log_scopes:
            vars.update({key: val for key, val in model.named_parameters()
                         if log_scope.match(key)})
        cls_logits, fg_feats, bg_feats = model(input)
        loss_cls = criterion(cls_logits, target)
        loss_pt = criterion2[0](bg_feats) + criterion2[2](fg_feats)
        loss_nt = criterion2[1](bg_feats, fg_feats)
        loss_conl = loss_pt + loss_nt
        loss = (1 - δ) * loss_conl + δ * loss_cls
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        prec1, prec5 = accuracy(cls_logits.data.contiguous(), target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))
        writer.add_scalar('loss_iter/train', loss.item(), update_train_step)
        writer.add_scalar('acc_iter/train_top1', prec1.item(), update_train_step)
        writer.add_scalar('acc_iter/train_top5', prec5.item(), update_train_step)

        for k, v in vars.items():
            writer.add_scalar(k, v.item(), update_train_step)

        if i % cfg.BASIC.DISP_FREQ == 0 or i == len(train_loader) - 1:
            print(('Train Epoch: [{0}][{1}/{2}],lr: {lr:.5f}\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                   'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i + 1, len(train_loader), loss=losses,
                top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr'])))
    return update_train_step, losses.avg, top1.avg, top5.avg


if __name__ == "__main__":
    main()
