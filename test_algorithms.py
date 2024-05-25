from Source.algorithms import iFairNMTF
from Source.algorithms import ind_fair_sc
from Source.algorithms import group_fair_sc
from Source.algorithms import normal_sc

import torch

import numpy as np
A = np.array([  [0, 1, 1, 0, 0, 0],
                [1, 0, 1, 0, 0, 0],
                [1, 1, 0, 1, 0, 0],
                [0, 0, 1, 0, 1, 1],
                [0, 0, 0, 1, 0, 1],
                [0, 0, 0, 1, 1, 0]], dtype=np.float32)
groups = np.array([0, 0, 1, 1, 1, 1])
k = 2
normalize_laplacian = False
print(normal_sc(A, k, normalize_laplacian))
print(group_fair_sc(A, groups, k, normalize_laplacian))
print(ind_fair_sc(A, groups, k, normalize_laplacian))
cluster = iFairNMTF(torch.from_numpy(A),
                torch.from_numpy(groups),
                k,
                100)
print(cluster)
