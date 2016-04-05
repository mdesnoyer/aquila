"""
Assembles the win matrix from the data, for training.
"""

from scipy import io
from scipy import sparse
import numpy as np

MPFN = 'task_data/aggregated_data/idx_2_id'
WDFN = 'task_data/aggregated_data/win_data.npy'
fnmap = dict()
print 'Loading index to filename map'
with open(MPFN, 'r') as f:
    for line in f:
        idx, fn = line.strip().split(',')
        fnmap[int(idx)] = fn + '.jpg'

WM = sparse.lil_matrix((len(fnmap.keys()), len(fnmap.keys())))
dat = np.load(WDFN)

WM[dat[:,0], dat[:,1]] = dat[:,2]
WM[dat[:,1], dat[:,0]] = dat[:,3]

with open('task_data/aggregated_data/win_matrix.mtx', 'w') as f:
    io.mmwrite(f, WM)