from glob import glob
import json
from dill import load

import numpy as np

from scipy import io
from scipy import sparse

from collections import defaultdict as ddict 

WIN_MATRIX_LOC = '/Users/davidlea/Desktop/testing/task_data/datasets/train/win_matrix.mtx'
COR_WIN_MATRIX_LOC = '/Users/davidlea/Desktop/testing/task_data/correct_win_matrix.mtx'
ALT_WIN_MATRIX_LOC = '/Users/davidlea/Desktop/testing/task_data/datasets/test/win_matrix.mtx'

FILE_MAP_LOC = '/Users/davidlea/Desktop/testing/task_data/aggregated_data/idx_2_id'
WIN_LIST_LOC = '/Users/davidlea/Desktop/testing/task_data/win_lists/newtask.csv'
LEG_WIN_LIST_LOC = '/Users/davidlea/Desktop/testing/task_data/win_lists/legacy.csv'

print 'Loading old win matrix'
wm1 = sparse.lil_matrix(io.mmread(WIN_MATRIX_LOC).astype(np.uint8))
wm2 = sparse.lil_matrix(io.mmread(ALT_WIN_MATRIX_LOC).astype(np.uint8))
old_win_matrix = wm1 + wm2

print 'Loading new win matrix'
new_win_matrix = sparse.lil_matrix(io.mmread(COR_WIN_MATRIX_LOC).astype(np.uint8))

fnmap = dict()
ifnmap = dict()
print 'Loading index to filename map'
with open(FILE_MAP_LOC, 'r') as f:
    for line in f:
        idx, fn = line.strip().split(',')
        fnmap[int(idx)] = fn + '.jpg'
        ifnmap[fn] = int(idx)

print 'compiling win matrix from file'
dimz = len(ifnmap)
file_win_matrix = sparse.lil_matrix((dimz, dimz)).astype(int)
targ = '/Users/davidlea/Desktop/testing/task_data/combined_win_data'
with open(targ, 'r') as f:
	for line in f:
		a, b, n = line.split(',')
		file_win_matrix[int(a), int(b)] = int(n)

# print 'compiling win matrix from id file'
# id_win_matrix = sparse.lil_matrix((dimz, dimz)).astype(int)
# targ = '/Users/davidlea/Desktop/testing/task_data/combined_win_data_fullids'
# with open(targ, 'r') as f:
# 	for line in f:
# 		a, b, n = line.split(',')
# 		id_win_matrix[ifnmap[a], ifnmap[b]] += int(n)

print 'old win matrix n elem:', old_win_matrix.sum()
print 'new win matrix n elem:', new_win_matrix.sum()
print 'file win matrix n elem:', file_win_matrix.sum()
# print 'id win matrix n elem:', id_win_matrix.sum()

# id_win_matrix and file_win_matrix are identical
#
# let's profile the source of errors

def get_indices(mtx1, mtx2):
	a1, b1 = mtx1.nonzero()
	a2, b2 = mtx2.nonzero()
	lsZ = list(set(zip(a1, b1) + zip(a2, b2)))
	a = [x[0] for x in lsZ]
	b = [x[1] for x in lsZ]
	return a, b

def get_vals(mtx, x_idx, y_idx):
	'''
	mtx is a matrix
	x_idx are the x indices
	y_idx are the y indices
	'''
	vals = mtx[x_idx, y_idx].A.squeeze()
	valsT = vals + mtx[y_idx, x_idx].A.squeeze()
	return vals, valsT

def profile_error(anom, targ):
	missing = 0
	introduced = 0
	transpose_err = 0
	corr = 0
	# extract all relevant values from the anom matrix
	print 'obtaining anomalous values'
	anomA, anomAT = get_vals(anom, *get_indices(anom, targ))
	print 'obtaining target values'
	targA, targAT = get_vals(targ, *get_indices(anom, targ))
	print 'Profiling errors'
	for idxN in range(len(anomA)):
		a1 = anomA[idxN]
		aT = anomAT[idxN]
		t1 = targA[idxN]
		tT = targAT[idxN]
		if a1 != t1:
			if aT == tT:
				transpose_err += abs(a1 - t1)
			elif a1 > t1:
				introduced += a1 - t1
			else:
				missing += t1 - a1
		else:
			corr += a1
	return missing, introduced, transpose_err, corr

m, i, t, c = profile_error(old_win_matrix, file_win_matrix)
print 'missing:',m,float(m)/(m+i+t+c),'introduced:',i,float(i)/(m+i+t+c),'transpose:',t,float(t)/(m+i+t+c),'correct',c,float(c)/(m+i+t+c)

m, i, t, c = profile_error(new_win_matrix, file_win_matrix)
print 'missing:',m,float(m)/(m+i+t+c),'introduced:',i,float(i)/(m+i+t+c),'transpose:',t,float(t)/(m+i+t+c),'correct',c,float(c)/(m+i+t+c)	


