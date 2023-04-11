from typing import Iterable

import torch
from torch import nn

import numpy as np

import copy
import itertools

from scipy.sparse.coo import coo_matrix
from scipy.sparse import diags


def inverse_schulz(X, iteration=5, alpha=0.002):
    """
    Computes an approximation of the matrix inversion using Newton-Schulz
    iterations
    Source NASA: https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19920002505.pdf
    """
    assert len(X.shape) >= 2, "Can't compute inverse on non-matrix"
    assert X.shape[-1] == X.shape[-2], "Must be batches of square matrices"

    with torch.no_grad():
        device = X.device
        eye = torch.eye(X.shape[-1], device=device)
        # alpha should be sufficiently small to have convergence
        inverse = alpha * torch.transpose(X, dim0=-2, dim1=-1)

    for _ in range(iteration):
        inverse = inverse @ (2 * eye - X @ inverse)

    return inverse


class SSFM(nn.Module):
    """

    ----------
    Parameters:
    - grid_size : tuple of ints
    A patched feature map shape to build with.
    e.g. [W1, ..., WN] where ：
    Wi - patch number of the axis i

    - num_connect : int
    the number of neighbor units to fuse against.

    - dilation : int
    the step size for fusion.

    - Tq_Mat: coo_matrix
    sparse coordinate matrix for adjacent positions relationship.
    Matrix is available automatically after the model initialization.
    Save and assign a matrix if the attention shape not change for reducing space cost.

    - Dq_Mat: coo_matrix
    sparse coordinate matrix for in-degree relationship.
    Matrix is available automatically after the model initialization.
    Save and assign a matrix if the attention shape not change for reducing space cost.

    - init_cfg (dict, optional): Initialization config dict. Default to None
    """

    def __init__(self,
                 grid_size: Iterable[int],
                 iteration: int,
                 num_connect: int = 8,
                 cross: int = 3,
                 Tq_Mat: coo_matrix = None,
                 Dq_Mat: coo_matrix = None,
                 LAq_Mat: coo_matrix = None,
                 loss_rate: float = 1,
                 init_cfg: dict = None, ):
        super(SSFM, self).__init__()

        self._grid_size = grid_size
        self.num_patch = np.prod(self.grid_size)
        self.dimension = len(grid_size)
        self.iteration = iteration
        self.cross = cross
        self._loss_rate = nn.Parameter(torch.ones([1]) * loss_rate)

        self._num_connect = num_connect

        self._Tq_Mat = Tq_Mat
        self._Dq_Mat = Dq_Mat
        self._LAq_Mat = LAq_Mat

        self._LAq_Mat= self.getLAq_Mat()
        self.init_weights()

    def getTq_mat(self) -> coo_matrix:
        if self._Tq_Mat is not None:
            return self._Tq_Mat
        # patch idx array
        idx_row = []
        idx_col = []

        for idx_r in range(self.num_patch):
            idxes = self.getDimIdx(idx_r)

            id_c1 = [[idx - 1, idx + 1]
                     for idx in idxes]
            id_c2 = [[idx - 2, idx + 2]
                     for idx in idxes]
            id_c3 = [[idx - 3, idx + 3]
                     for idx in idxes]
            id_c4 = [[idx - 4, idx + 4]
                     for idx in idxes]
            id_c5 = [[idx - 5, idx + 5]
                     for idx in idxes]
            id_c6 = [[idx - 6, idx + 6]
                     for idx in idxes]

            id_c = self.selectValidIdx(self.connect(idxes, id_c1, id_c2, id_c3, id_c4, id_c5, id_c6))
            idp_c = [self.getPatchIdx(i) for i in id_c]

            for idx_c in idp_c:
                idx_row.append(idx_r)
                idx_col.append(idx_c)

        self._Tq_Mat = coo_matrix((np.full(len(idx_row), 1, dtype=int), (idx_row, idx_col)), shape=(
            self.num_patch, self.num_patch), dtype=int)  # Dl

        return self._Tq_Mat

    def getDq_Mat(self):
        '''
        Get in-degree matrix. Row is the in-direction.
        '''
        if self._Dq_Mat is not None:
            return self._Dq_Mat
        self._Tq_Mat = self.getAdj()
        self._Dq_Mat = diags(self._Tq_Mat.sum(axis=0).A1)  # Al
        return self._Dq_Mat

    def getLAq_Mat(self):
        if self.LAq_Mat is not None:
            return self.LAq_Mat
        if self.Tq_Mat is None:
            self.Tq_Mat = self.getAdj()
        if self.Dq_Mat is None:
            self.Dq_Mat = self.getDq_Mat()
        LAq_Mat = self.Dq_Mat - self.Tq_Mat  # Li
        return LAq_Mat

    def getPatchIdx(self, idxes):  # [[12, 13], [13, 12]]
        '''
        Get patch index by dimension indexes, e.g. idx_patch_D_W_H (2, 10, 9) = 2*s1*s2 + 10*s2 + 9
        where s0, s1, s2 is the maximum number of patches along axies.

        Without access and output safety check
        '''
        idx_axis = [idxes[i] * np.prod(self._grid_size[1 + i:])
                    for i in range(self.dimension)]
        return np.sum(idx_axis, dtype=int)

    def getDimIdx(self, idx):
        '''
        Get dimension index by patch idx, e.g. idx_dim_D_W_H (10) = 10/(s1*s2), left/s2, left'
        where s0, s1, s2 is the maximum number of patches along axies.

        Without access and output safety check
        '''
        left = idx
        idxes = []
        for i in range(self.dimension):
            prod_ = np.prod(self._grid_size[1 + i:], dtype=int)
            idx_ = left // prod_
            left -= idx_ * prod_
            idxes.append(idx_)
        return idxes

    def connect(self, idxes, id_c1, id_c2, id_c3, id_c4, id_c5, id_c6):

        res = []
        around3 = []
        around4 = []
        around5 = []
        around6 = []
        if self.cross == 1:
            for i, ref in enumerate(id_c1):
                idxes_1 = copy.deepcopy(idxes)
                idxes_1[i] = ref[0]
                idxes_2 = copy.deepcopy(idxes)
                idxes_2[i] = ref[1]
                res.append(idxes_1)
                res.append(idxes_2)
            idc1 = list(itertools.product(*id_c1))
            for i, ref in enumerate(idc1):
                ind = [1, 2]
                ind[0] = ref[0]
                ind[1] = ref[1]
                res.append(ind)

        elif self.cross == 2:
            for i, ref in enumerate(id_c1):
                idxes_1 = copy.deepcopy(idxes)
                idxes_1[i] = ref[0]
                idxes_2 = copy.deepcopy(idxes)
                idxes_2[i] = ref[1]
                res.append(idxes_1)
                res.append(idxes_2)
            for i, ref in enumerate(id_c2):
                idxes_1 = copy.deepcopy(idxes)
                idxes_1[i] = ref[0]
                idxes_2 = copy.deepcopy(idxes)
                idxes_2[i] = ref[1]
                res.append(idxes_1)
                res.append(idxes_2)
            idc1 = list(itertools.product(*id_c1))
            idc2 = list(itertools.product(*id_c2))
            for i, ref in enumerate(idc1):
                ind = [1, 2]
                ind[0] = ref[0]
                ind[1] = ref[1]
                res.append(ind)
            for i, ref in enumerate(idc2):
                ind = [1, 2]
                ind[0] = ref[0]
                ind[1] = ref[1]
                res.append(ind)
            for i, j in zip(idc1, idc2):
                c = [1, 2]
                d = [1, 2]
                c[0] = i[0]
                c[1] = j[1]
                res.append(c)
                d[1] = i[1]
                d[0] = j[0]
                res.append(d)
        elif self.cross == 3:
            for i, ref in enumerate(id_c1):
                idxes_1 = copy.deepcopy(idxes)
                idxes_1[i] = ref[0]
                idxes_2 = copy.deepcopy(idxes)
                idxes_2[i] = ref[1]
                res.append(idxes_1)
                res.append(idxes_2)
            for i, ref in enumerate(id_c2):
                idxes_1 = copy.deepcopy(idxes)
                idxes_1[i] = ref[0]
                idxes_2 = copy.deepcopy(idxes)
                idxes_2[i] = ref[1]
                res.append(idxes_1)
                res.append(idxes_2)
            for i, ref in enumerate(id_c3):
                idxes_1 = copy.deepcopy(idxes)
                idxes_1[i] = ref[0]
                idxes_2 = copy.deepcopy(idxes)
                idxes_2[i] = ref[1]
                around3.append(idxes_1)
                around3.append(idxes_2)
                res.append(idxes_1)
                res.append(idxes_2)
            idc1 = list(itertools.product(*id_c1))
            idc2 = list(itertools.product(*id_c2))
            idc3 = list(itertools.product(*id_c3))
            for i, ref in enumerate(idc1):
                ind = [1, 2]
                ind[0] = ref[0]
                ind[1] = ref[1]
                res.append(ind)
            for i, ref in enumerate(idc2):
                ind = [1, 2]
                ind[0] = ref[0]
                ind[1] = ref[1]
                res.append(ind)
            for i, ref in enumerate(idc3):
                ind = [1, 2]
                ind[0] = ref[0]
                ind[1] = ref[1]
                res.append(ind)
            for i, j in zip(idc1, idc2):
                c = [1, 2]
                d = [1, 2]
                c[0] = i[0]
                c[1] = j[1]
                res.append(c)
                print('1', c)
                d[1] = i[1]
                d[0] = j[0]
                res.append(d)
            for i, j in zip(idc2, idc3):
                c = [1, 2]
                d = [1, 2]
                c[0] = i[0]
                c[1] = j[1]
                res.append(c)
                print('1', c)
                d[1] = i[1]
                d[0] = j[0]
                print('2', d)
                res.append(d)
            for j, i in enumerate(around3):
                if j <= 1:
                    c = [1, 2]
                    d = [1, 2]
                    c[0] = i[0]
                    c[1] = i[1] + 1
                    d[0] = i[0]
                    d[1] = i[1] - 1
                    res.append(c)
                    res.append(d)
                else:
                    c = [1, 2]
                    d = [1, 2]
                    c[0] = i[0] + 1
                    c[1] = i[1]
                    d[0] = i[0] - 1
                    d[1] = i[1]
                    res.append((c))
                    res.append(d)
        elif self.cross ==4:
            for i, ref in enumerate(id_c1):
                idxes_1 = copy.deepcopy(idxes)
                idxes_1[i] = ref[0]
                idxes_2 = copy.deepcopy(idxes)
                idxes_2[i] = ref[1]
                res.append(idxes_1)
                res.append(idxes_2)
            for i, ref in enumerate(id_c2):
                idxes_1 = copy.deepcopy(idxes)
                idxes_1[i] = ref[0]
                idxes_2 = copy.deepcopy(idxes)
                idxes_2[i] = ref[1]
                res.append(idxes_1)
                res.append(idxes_2)
            for i, ref in enumerate(id_c3):
                idxes_1 = copy.deepcopy(idxes)
                idxes_1[i] = ref[0]
                idxes_2 = copy.deepcopy(idxes)
                idxes_2[i] = ref[1]
                around3.append(idxes_1)
                around3.append(idxes_2)
                res.append(idxes_1)
                res.append(idxes_2)
            for i, ref in enumerate(id_c4):
                idxes_1 = copy.deepcopy(idxes)
                idxes_1[i] = ref[0]
                idxes_2 = copy.deepcopy(idxes)
                idxes_2[i] = ref[1]
                around4.append(idxes_1)
                around4.append(idxes_2)
                res.append(idxes_1)
                res.append(idxes_2)
            idc1 = list(itertools.product(*id_c1))
            idc2 = list(itertools.product(*id_c2))
            idc3 = list(itertools.product(*id_c3))
            idc4 = list(itertools.product(*id_c4))
            for i, ref in enumerate(idc1):
                ind = [1, 2]
                ind[0] = ref[0]
                ind[1] = ref[1]
                res.append(ind)
            for i, ref in enumerate(idc2):
                ind = [1, 2]
                ind[0] = ref[0]
                ind[1] = ref[1]
                res.append(ind)
            for i, ref in enumerate(idc3):
                ind = [1, 2]
                ind[0] = ref[0]
                ind[1] = ref[1]
                res.append(ind)
            for i, ref in enumerate(idc4):
                ind = [1, 2]
                ind[0] = ref[0]
                ind[1] = ref[1]
                res.append(ind)
            for i, j in zip(idc1, idc2):
                c = [1, 2]
                d = [1, 2]
                c[0] = i[0]
                c[1] = j[1]
                res.append(c)
                print('1', c)
                d[1] = i[1]
                d[0] = j[0]
                res.append(d)
            for i, j in zip(idc2, idc3):
                c = [1, 2]
                d = [1, 2]
                c[0] = i[0]
                c[1] = j[1]
                res.append(c)
                print('1', c)
                d[1] = i[1]
                d[0] = j[0]
                print('2', d)
                res.append(d)
            for i, j in zip(idc3, idc4):
                c = [1, 2]
                d = [1, 2]
                c[0] = i[0]
                c[1] = j[1]
                res.append(c)
                d[1] = i[1]
                d[0] = j[0]
                res.append(d)
            for j, i in enumerate(around3):
                if j <= 1:
                    c = [1, 2]
                    d = [1, 2]
                    c[0] = i[0]
                    c[1] = i[1] + 1
                    d[0] = i[0]
                    d[1] = i[1] - 1
                    res.append(c)
                    res.append(d)
                else:
                    c = [1, 2]
                    d = [1, 2]
                    c[0] = i[0] + 1
                    c[1] = i[1]
                    d[0] = i[0] - 1
                    d[1] = i[1]
                    res.append((c))
                    res.append(d)
            for j, i in enumerate(around4):
                if j <= 1:
                    for k in [1, 2]:
                        c = [1, 2]
                        d = [1, 2]
                        c[0] = i[0]
                        c[1] = i[1] + k
                        d[0] = i[0]
                        d[1] = i[1] - k
                        res.append(c)
                        res.append(d)
                else:
                    for l in [1, 2]:
                        c = [1, 2]
                        d = [1, 2]
                        c[0] = i[0] + l
                        c[1] = i[1]
                        d[0] = i[0] - l
                        d[1] = i[1]
                        res.append((c))
                        res.append(d)
        elif self.cross ==5:
            for i, ref in enumerate(id_c1):
                idxes_1 = copy.deepcopy(idxes)
                idxes_1[i] = ref[0]
                idxes_2 = copy.deepcopy(idxes)
                idxes_2[i] = ref[1]
                res.append(idxes_1)
                res.append(idxes_2)
            for i, ref in enumerate(id_c2):
                idxes_1 = copy.deepcopy(idxes)
                idxes_1[i] = ref[0]
                idxes_2 = copy.deepcopy(idxes)
                idxes_2[i] = ref[1]
                res.append(idxes_1)
                res.append(idxes_2)
            for i, ref in enumerate(id_c3):
                idxes_1 = copy.deepcopy(idxes)
                idxes_1[i] = ref[0]
                idxes_2 = copy.deepcopy(idxes)
                idxes_2[i] = ref[1]
                around3.append(idxes_1)
                around3.append(idxes_2)
                res.append(idxes_1)
                res.append(idxes_2)
            for i, ref in enumerate(id_c4):
                idxes_1 = copy.deepcopy(idxes)
                idxes_1[i] = ref[0]
                idxes_2 = copy.deepcopy(idxes)
                idxes_2[i] = ref[1]
                around4.append(idxes_1)
                around4.append(idxes_2)
                res.append(idxes_1)
                res.append(idxes_2)
            for i, ref in enumerate(id_c5):
                idxes_1 = copy.deepcopy(idxes)
                idxes_1[i] = ref[0]
                idxes_2 = copy.deepcopy(idxes)
                idxes_2[i] = ref[1]
                around5.append(idxes_1)
                around5.append(idxes_2)
                res.append(idxes_1)
                res.append(idxes_2)
            idc1 = list(itertools.product(*id_c1))
            idc2 = list(itertools.product(*id_c2))
            idc3 = list(itertools.product(*id_c3))
            idc4 = list(itertools.product(*id_c4))
            idc5 = list(itertools.product(*id_c5))
            for i, ref in enumerate(idc1):
                ind = [1, 2]
                ind[0] = ref[0]
                ind[1] = ref[1]
                res.append(ind)
            for i, ref in enumerate(idc2):
                ind = [1, 2]
                ind[0] = ref[0]
                ind[1] = ref[1]
                res.append(ind)
            for i, ref in enumerate(idc3):
                ind = [1, 2]
                ind[0] = ref[0]
                ind[1] = ref[1]
                res.append(ind)
            for i, ref in enumerate(idc4):
                ind = [1, 2]
                ind[0] = ref[0]
                ind[1] = ref[1]
                res.append(ind)
            for i, ref in enumerate(idc5):
                ind = [1, 2]
                ind[0] = ref[0]
                ind[1] = ref[1]
                res.append(ind)
            for i, j in zip(idc1, idc2):
                c = [1, 2]
                d = [1, 2]
                c[0] = i[0]
                c[1] = j[1]
                res.append(c)
                print('1', c)
                d[1] = i[1]
                d[0] = j[0]
                res.append(d)
            for i, j in zip(idc2, idc3):
                c = [1, 2]
                d = [1, 2]
                c[0] = i[0]
                c[1] = j[1]
                res.append(c)
                print('1', c)
                d[1] = i[1]
                d[0] = j[0]
                print('2', d)
                res.append(d)
            for i, j in zip(idc3, idc4):
                c = [1, 2]
                d = [1, 2]
                c[0] = i[0]
                c[1] = j[1]
                res.append(c)
                d[1] = i[1]
                d[0] = j[0]
                res.append(d)
            for i, j in zip(idc3, idc5):
                c = [1, 2]
                d = [1, 2]
                c[0] = i[0]
                c[1] = j[1]
                res.append(c)
                d[1] = i[1]
                d[0] = j[0]
                res.append(d)
            for i, j in zip(idc4, idc5):
                c = [1, 2]
                d = [1, 2]
                c[0] = i[0]
                c[1] = j[1]
                res.append(c)
                d[1] = i[1]
                d[0] = j[0]
                res.append(d)
            for j, i in enumerate(around3):
                if j <= 1:
                    c = [1, 2]
                    d = [1, 2]
                    c[0] = i[0]
                    c[1] = i[1] + 1
                    d[0] = i[0]
                    d[1] = i[1] - 1
                    res.append(c)
                    res.append(d)
                else:
                    c = [1, 2]
                    d = [1, 2]
                    c[0] = i[0] + 1
                    c[1] = i[1]
                    d[0] = i[0] - 1
                    d[1] = i[1]
                    res.append((c))
                    res.append(d)
            for j, i in enumerate(around4):
                if j <= 1:
                    for k in [1, 2]:
                        c = [1, 2]
                        d = [1, 2]
                        c[0] = i[0]
                        c[1] = i[1] + k
                        d[0] = i[0]
                        d[1] = i[1] - k
                        res.append(c)
                        res.append(d)
                else:
                    for l in [1, 2]:
                        c = [1, 2]
                        d = [1, 2]
                        c[0] = i[0] + l
                        c[1] = i[1]
                        d[0] = i[0] - l
                        d[1] = i[1]
                        res.append((c))
                        res.append(d)
            for j, i in enumerate(around5):
                if j <= 1:
                    for k in [1, 2]:
                        c = [1, 2]
                        d = [1, 2]
                        c[0] = i[0]
                        c[1] = i[1] + k
                        d[0] = i[0]
                        d[1] = i[1] - k
                        res.append(c)
                        res.append(d)
                else:
                    for l in [1, 2]:
                        c = [1, 2]
                        d = [1, 2]
                        c[0] = i[0] + l
                        c[1] = i[1]
                        d[0] = i[0] - l
                        d[1] = i[1]
                        res.append((c))
                        res.append(d)
        elif self.cross ==6:
            for i, ref in enumerate(id_c1):
                idxes_1 = copy.deepcopy(idxes)
                idxes_1[i] = ref[0]
                idxes_2 = copy.deepcopy(idxes)
                idxes_2[i] = ref[1]
                res.append(idxes_1)
                res.append(idxes_2)
            for i, ref in enumerate(id_c2):
                idxes_1 = copy.deepcopy(idxes)
                idxes_1[i] = ref[0]
                idxes_2 = copy.deepcopy(idxes)
                idxes_2[i] = ref[1]
                res.append(idxes_1)
                res.append(idxes_2)
            for i, ref in enumerate(id_c3):
                idxes_1 = copy.deepcopy(idxes)
                idxes_1[i] = ref[0]
                idxes_2 = copy.deepcopy(idxes)
                idxes_2[i] = ref[1]
                around3.append(idxes_1)
                around3.append(idxes_2)
                res.append(idxes_1)
                res.append(idxes_2)
            for i, ref in enumerate(id_c4):
                idxes_1 = copy.deepcopy(idxes)
                idxes_1[i] = ref[0]
                idxes_2 = copy.deepcopy(idxes)
                idxes_2[i] = ref[1]
                around4.append(idxes_1)
                around4.append(idxes_2)
                res.append(idxes_1)
                res.append(idxes_2)
            for i, ref in enumerate(id_c5):
                idxes_1 = copy.deepcopy(idxes)
                idxes_1[i] = ref[0]
                idxes_2 = copy.deepcopy(idxes)
                idxes_2[i] = ref[1]
                around5.append(idxes_1)
                around5.append(idxes_2)
                res.append(idxes_1)
                res.append(idxes_2)
            for i, ref in enumerate(id_c6):
                idxes_1 = copy.deepcopy(idxes)
                idxes_1[i] = ref[0]
                idxes_2 = copy.deepcopy(idxes)
                idxes_2[i] = ref[1]
                around5.append(idxes_1)
                around5.append(idxes_2)
                res.append(idxes_1)
                res.append(idxes_2)
            idc1 = list(itertools.product(*id_c1))
            idc2 = list(itertools.product(*id_c2))
            idc3 = list(itertools.product(*id_c3))
            idc4 = list(itertools.product(*id_c4))
            idc5 = list(itertools.product(*id_c5))
            idc6 = list(itertools.product(*id_c6))
            for i, ref in enumerate(idc1):
                ind = [1, 2]
                ind[0] = ref[0]
                ind[1] = ref[1]
                res.append(ind)
            for i, ref in enumerate(idc2):
                ind = [1, 2]
                ind[0] = ref[0]
                ind[1] = ref[1]
                res.append(ind)
            for i, ref in enumerate(idc3):
                ind = [1, 2]
                ind[0] = ref[0]
                ind[1] = ref[1]
                res.append(ind)
            for i, ref in enumerate(idc4):
                ind = [1, 2]
                ind[0] = ref[0]
                ind[1] = ref[1]
                res.append(ind)
            for i, ref in enumerate(idc5):
                ind = [1, 2]
                ind[0] = ref[0]
                ind[1] = ref[1]
                res.append(ind)
            for i, ref in enumerate(idc6):
                ind = [1, 2]
                ind[0] = ref[0]
                ind[1] = ref[1]
                res.append(ind)
            for i, j in zip(idc1, idc2):
                c = [1, 2]
                d = [1, 2]
                c[0] = i[0]
                c[1] = j[1]
                res.append(c)
                print('1', c)
                d[1] = i[1]
                d[0] = j[0]
                res.append(d)
            for i, j in zip(idc2, idc3):
                c = [1, 2]
                d = [1, 2]
                c[0] = i[0]
                c[1] = j[1]
                res.append(c)
                print('1', c)
                d[1] = i[1]
                d[0] = j[0]
                print('2', d)
                res.append(d)
            for i, j in zip(idc3, idc4):
                c = [1, 2]
                d = [1, 2]
                c[0] = i[0]
                c[1] = j[1]
                res.append(c)
                d[1] = i[1]
                d[0] = j[0]
                res.append(d)
            for i, j in zip(idc3, idc5):
                c = [1, 2]
                d = [1, 2]
                c[0] = i[0]
                c[1] = j[1]
                res.append(c)
                d[1] = i[1]
                d[0] = j[0]
                res.append(d)
            for i, j in zip(idc4, idc5):
                c = [1, 2]
                d = [1, 2]
                c[0] = i[0]
                c[1] = j[1]
                res.append(c)
                d[1] = i[1]
                d[0] = j[0]
                res.append(d)
            for i, j in zip(idc5, idc6):
                c = [1, 2]
                d = [1, 2]
                c[0] = i[0]
                c[1] = j[1]
                res.append(c)
                d[1] = i[1]
                d[0] = j[0]
                res.append(d)
            for i, j in zip(idc4, idc6):
                c = [1, 2]
                d = [1, 2]
                c[0] = i[0]
                c[1] = j[1]
                res.append(c)
                d[1] = i[1]
                d[0] = j[0]
                res.append(d)
            for i, j in zip(idc3, idc6):
                c = [1, 2]
                d = [1, 2]
                c[0] = i[0]
                c[1] = j[1]
                res.append(c)
                d[1] = i[1]
                d[0] = j[0]
                res.append(d)
            for j, i in enumerate(around3):
                if j <= 1:
                    c = [1, 2]
                    d = [1, 2]
                    c[0] = i[0]
                    c[1] = i[1] + 1
                    d[0] = i[0]
                    d[1] = i[1] - 1
                    res.append(c)
                    res.append(d)
                else:
                    c = [1, 2]
                    d = [1, 2]
                    c[0] = i[0] + 1
                    c[1] = i[1]
                    d[0] = i[0] - 1
                    d[1] = i[1]
                    res.append((c))
                    res.append(d)
            for j, i in enumerate(around4):
                if j <= 1:
                    for k in [1, 2]:
                        c = [1, 2]
                        d = [1, 2]
                        c[0] = i[0]
                        c[1] = i[1] + k
                        d[0] = i[0]
                        d[1] = i[1] - k
                        res.append(c)
                        res.append(d)
                else:
                    for l in [1, 2]:
                        c = [1, 2]
                        d = [1, 2]
                        c[0] = i[0] + l
                        c[1] = i[1]
                        d[0] = i[0] - l
                        d[1] = i[1]
                        res.append(c)
                        res.append(d)
            for j, i in enumerate(around5):
                if j <= 1:
                    for k in [1, 2]:
                        c = [1, 2]
                        d = [1, 2]
                        c[0] = i[0]
                        c[1] = i[1] + k
                        d[0] = i[0]
                        d[1] = i[1] - k
                        res.append(c)
                        res.append(d)
                else:
                    for l in [1, 2]:
                        c = [1, 2]
                        d = [1, 2]
                        c[0] = i[0] + l
                        c[1] = i[1]
                        d[0] = i[0] - l
                        d[1] = i[1]
                        res.append(c)
                        res.append(d)
            for j, i in enumerate(around6):
                if j <= 1:
                    for k in [1, 2]:
                        c = [1, 2]
                        d = [1, 2]
                        c[0] = i[0]
                        c[1] = i[1] + k
                        d[0] = i[0]
                        d[1] = i[1] - k
                        res.append(c)
                        res.append(d)
                else:
                    for l in [1, 2]:
                        c = [1, 2]
                        d = [1, 2]
                        c[0] = i[0] + l
                        c[1] = i[1]
                        d[0] = i[0] - l
                        d[1] = i[1]
                        res.append(c)
                        res.append(d)
        return res

    def selectValidIdx(self, idxes):
        res = list(zip(*idxes))
        sel = np.full(len(idxes), True, dtype=bool)
        for i, idx in enumerate(res):
            a = np.array(idx)
            sel = sel & (a >= 0) & (a < self._grid_size[i])

        return np.array(idxes)[sel]

    @property
    def num_connect(self):
        return self._num_connect

    @num_connect.setter
    def num_connect(self, val):
        self._num_connect = val

    @property
    def dilation(self):
        return self._dilation

    @dilation.setter
    def dilation(self, val):
        self._dilation = val

    @property
    def grid_size(self):
        return self._grid_size

    @property
    def Tq_Mat(self):
        return self._Tq_Mat

    @Tq_Mat.setter
    def Tq_Mat(self, val):
        self._Tq_Mat = val

    @property
    def Dq_Mat(self):
        return self._Dq_Mat

    @Dq_Mat.setter
    def Dq_Mat(self, val):
        self._Dq_Mat = val

    @property
    def LAq_Mat(self):
        return self._lap

    @LAq_Mat.setter
    def LAq_Mat(self, val):
        self._lap = val

    @property
    def loss_rate(self):
        return self._loss_rate

    @loss_rate.setter
    def loss_rate(self, val):
        self._loss_rate = nn.Parameter(torch.empty([1]) * val)

    def forward(self, sim):
        r"""Allows the model to generate the fusion-based attention matrix.
        Fuse one time -> one iteration only.
    Args:
        sim: patch pair wise similarity matrix.

        Shapes for inputs:
        - sim: :math:`(B, N, N)`, where B is the batch size, N is the target `spatial` sequence length.
        Shapes for outputs:
        - fAttn_output: :math:`(B, N, N)` where B is the batch size, N is the target `spatial` sequence length.

    Examples:

        # >>> d
    """
        if len(sim.shape) != 3:
            raise ValueError(
                f'Expect the patch pair-wise similarity matrix\'s shape to be [B, N, N], but got {sim.shape} instead.'
            )
        assert sim.shape[-1] == self.num_patch and sim.shape[-1] == sim.shape[
            -2], f'Expect he patch pair-wise similarity matrix to have {self.num_patch} tokens, but got {sim.shape[-1]}.'

        # TODO: test the module functionality
        with torch.no_grad():
            factory_kwargs = {'device': sim.device, 'dtype': sim.dtype}
            Fq = torch.tensor(self.LAq_Mat.todense().A, **factory_kwargs)
            # lr = torch.sigmoid(self.loss_rate.to(factory_kwargs['device']))
            lr = self.loss_rate.to(factory_kwargs['device'])

        Fq = torch.mul(Fq, lr * sim - 1)
        Fq = inverse_schulz(Fq, iteration=self.iteration)
        Fq = Fq.transpose(dim0=-2, dim1=-1)

        return Fq

    def init_weights(self):

        pass
        # nn.init.constant_(self.loss_rate, 0)

    def __repr__(self):
        s = super().__repr__()
        s = s[:-2]
        s += '\n  fusion_cfg:('
        s += f'\n    grid_size={self.grid_size}'
        s += f'\n    dilation={self.dilation}'
        s += f'\n    num_connect={self.num_connect}'
        s += f'\n    loss_rate={self.loss_rate}'
        s += '\n))'
        return s
