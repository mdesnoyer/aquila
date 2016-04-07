"""
Attempting to validate the data that we have in the win matrix.
"""

from glob import glob
import json
from dill import load

import numpy as np

from scipy import io
from scipy import sparse

from collections import defaultdict as ddict 

FILE_MAP_LOC = '/Users/davidlea/Desktop/testing/task_data/aggregated_data/idx_2_id'
WIN_MATRIX_LOC = '/Users/davidlea/Desktop/testing/task_data/datasets/train/win_matrix.mtx'
ALT_WIN_MATRIX_LOC = '/Users/davidlea/Desktop/testing/task_data/datasets/test/win_matrix.mtx'
WIN_LIST_LOC = '/Users/davidlea/Desktop/testing/task_data/win_lists/newtask.csv'
LEG_WIN_LIST_LOC = '/Users/davidlea/Desktop/testing/task_data/win_lists/legacy.csv'

fnmap = dict()
ifnmap = dict()
print 'Loading index to filename map'
with open(FILE_MAP_LOC, 'r') as f:
    for line in f:
        idx, fn = line.strip().split(',')
        fnmap[int(idx)] = fn + '.jpg'
        ifnmap[fn] = int(idx)

print 'Reading the (new) win list'
win_tab = ddict(lambda: ddict(lambda: 0))
tot_wins_obs_in_wlist = 0
tot_wins_obs_in_both_wlist = 0
with open(WIN_LIST_LOC, 'r') as f:
	for line in f:
		a, b, n = line.split(',')
		win_tab[a][b] += int(n)
		tot_wins_obs_in_wlist += int(n)
		tot_wins_obs_in_both_wlist += int(n)

print 'Reading the (legacy) win list'
with open(LEG_WIN_LIST_LOC, 'r') as f:
	for line in f:
		a, b, nab, nba = line.split(',')
		a = a.split('.')[0]
		b = b.split('.')[0]
		if a not in ifnmap:
			continue
		if b not in ifnmap:
			continue
		win_tab[a][b] += int(nab)
		tot_wins_obs_in_both_wlist += int(n)
		win_tab[b][a] += int(nba)
		tot_wins_obs_in_both_wlist += int(n)

print 'Loading win matrix'
wm1 = sparse.lil_matrix(io.mmread(WIN_MATRIX_LOC).astype(np.uint8))
wm2 = sparse.lil_matrix(io.mmread(ALT_WIN_MATRIX_LOC).astype(np.uint8))
win_matrix = wm1 + wm2

print 'fetching file list'
files = glob('/Users/davidlea/Desktop/testing/mturk_task_json/t_*')


def read_file(filen):
	with open(filen, 'r') as f:
		fdict = load(f)
		task_json = json.loads(fdict['completion_data:response_json'])
		blocks = [x for x in task_json if x['trial_type'] == 'click-choice']
		choices = fdict['completion_data:choices']
		return blocks, choices

def parse_blocks(blocks):
	# for each block two items:
	# [WIN 1, LOSE 1, WIN 2, LOSE 2, CHOICE_TYPE, CHOICE, IMAGES]
	data = []
	for block in blocks:
		choice_type = block['action_type']
		images = [x['id'] for x in block['stims']]
		choice = block['choice']
		rt = block['rt']
		if rt < 400:
			continue
		if choice == -1:
			continue
		unchose = [x for x in images if x != choice]
		if choice_type == 'keep':
			win1 = choice
			win2 = choice
			lose1 = unchose[0]
			lose2 = unchose[1]
		elif choice_type == 'reject':
			lose1 = choice
			lose2 = choice
			win1 = unchose[0]
			win2 = unchose[1]
		else:
			print 'error...'
			continue
		cdat = [win1, lose1, win2, lose2, choice_type, choice]
		data.append(cdat)
	return data

def parse_to_idx(parse):
	'''Takes the parsed block and converts all IDs to indices'''
	ndata = []
	for win1, lose1, win2, lose2, choice_type, choice in parse:
		ndata.append([ifnmap[win1], ifnmap[lose1],
					  ifnmap[win2], ifnmap[lose2],
					  choice_type, ifnmap[choice]])
	return ndata

# idicates when the win matrix appears to be incorrect
def validate_idx_parse(idx_parse):
	import ipdb
	err = 0
	for w1, l1, w2, l2, ctype, cchoice in idx_parse:
		if not win_matrix[w1, l1]:
			err += 1
		if not win_matrix[w2, l2]:
			err += 1
	return err == 0, err

# indicates when the win list appears to be incorrect
def validate_orig_parse(parse):
	err = 0
	for w1, l1, w2, l2, ctype, cchoice in parse:
		if not win_tab[w1][l1]:
			err += 1
		if not win_tab[w2][l2]:
			err += 1
	return err == 0, err

wins_obs_in_json = 0
tot_wins_obs_in_wm = win_matrix.sum() #np.sum(win_matrix)
tot_wins_obs_in_wlist = tot_wins_obs_in_wlist
tot_wins_obs_in_both_wlist = tot_wins_obs_in_both_wlist

tot_err_in_win_list = 0
for n, cfile in enumerate(files):
	if not n % 100:
		print '%i / %i: obs err in WM: %i' % (n, len(files), tot_err_in_win_list)
	blocks, choices = read_file(cfile)
	dat = parse_blocks(blocks)
	wins_obs_in_json += (len(dat) * 2)
	idx_dat = parse_to_idx(dat)
	# print 'unparsed errors', validate_orig_parse(dat)
	xx, xy = validate_idx_parse(idx_dat)
	tot_err_in_win_list += xy
	# print 'parsed errors', validate_idx_parse(idx_dat)

# holy fuck it looks like there's an error in the win matrix
# fuuuuuuuck
# this explains why training failed so hard.
print '[should match]', tot_wins_obs_in_wlist, wins_obs_in_json
print '[should match]', tot_wins_obs_in_wm, tot_wins_obs_in_both_wlist
