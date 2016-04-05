"""
This loads the win data matrix, test.mtx, and identifies singletons--i.e.,
images that only have one edge. These will be used as our training and test
data.
"""

from scipy import io
from scipy import sparse
import numpy as np

MPFN = 'task_data/aggregated_data/idx_2_id'
fnmap = dict()
print 'Loading index to filename map'
with open(MPFN, 'r') as f:
    for line in f:
        idx, fn = line.strip().split(',')
        fnmap[int(idx)] = fn + '.jpg'

print 'Loading win matrix'
x = sparse.lil_matrix(io.mmread('task_data/aggregated_data/win_matrix.mtx'))
print 'Complete, transposing x'
xT = x.transpose()
x_ax_sum = np.squeeze(np.array(x.sum(axis=0)))
y_ax_sum = np.squeeze(np.array(x.sum(axis=1)))
singleton_idxs = (np.array(x_ax_sum + y_ax_sum) == 1).nonzero()[0]
singleton_ids = [fnmap[sidx] for sidx in singleton_idxs]
print 'Found %i singletons' % len(singleton_idxs)
print 'Acquiring indices'
s_idx1 = []
s_idx2 = []
for n, xxx in enumerate(singleton_idxs):
    if not n % 100:
        print n
    qqq = list(x[xxx].nonzero()[1])
    if qqq:
        s_idx1.append(xxx)
        s_idx2 += qqq
        #continue
    qqq = list(xT[xxx].nonzero()[1])
    if qqq:
        s_idx1 += qqq
        s_idx2.append(xxx)
testing_win_mtx = sparse.lil_matrix((x.shape))
training_win_mtx = x.copy()
all_ids = []
print 'Creating test / train matrices'
for idxa, idxb in zip(s_idx1, s_idx2):
    testing_win_mtx[idxa, idxb] = x[idxa, idxb]
    testing_win_mtx[idxb, idxa] = x[idxb, idxa]
    training_win_mtx[idxa, idxb] = 0
    training_win_mtx[idxb, idxa] = 0
    all_ids.append(fnmap[idxa])
    all_ids.append(fnmap[idxb])
io.mmwrite('task_data/datasets/train/win_matrix.mtx', training_win_mtx)
io.mmwrite('task_data/datasets/test/win_matrix.mtx', testing_win_mtx)
