"""
We need to assemble some new win matrices based the fact that
the current one was incorrectly generated!

We will generate the win matrix from the JSON data as well as
the legacy win lists.
"""

from glob import glob
import json
from dill import load, dump

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

print 'Reading the (legacy) win list'
win_tab = ddict(lambda: ddict(lambda: 0))
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
		# nba is redundant.

print 'fetching file list'
files = glob('/Users/davidlea/Desktop/testing/mturk_task_json/t_*')

def read_file(filen):
	with open(filen, 'r') as f:
		fdict = load(f)
		task_json = json.loads(fdict['completion_data:response_json'])
		blocks = [x for x in task_json if x['trial_type'] == 'click-choice']
		choices = fdict['completion_data:choices']
		# return blocks, choices
		return blocks

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

def update_win_table_from_parse(parse):
	for win1, lose1, win2, lose2, choice_type, choice in parse:
		win_tab[win1][lose1] += 1
		win_tab[win2][lose2] += 1

print 'Wins in win table:', sum([sum(x.values()) for k, x in win_tab.iteritems()])
print 'updating win table with new information'
for n, cfile in enumerate(files):
	if not n % 100:
		print '%i / %i' % (n, len(files))
	blocks = read_file(cfile)
	dat = parse_blocks(blocks)
	update_win_table_from_parse(dat)
# win_matrix = sparse.lil_matrix((len(ifnmap, ifnmap)))
print 'Final wins in win table:', sum([sum(x.values()) for k, x in win_tab.iteritems()])

print 'Creating win table'
tot = 0
dimz = len(ifnmap)
win_matrix = sparse.lil_matrix((dimz, dimz))
for wnr, v in win_tab.iteritems():
	for lsr, n in v.iteritems():
		if not tot % 1000:
			print tot
		win_matrix[ifnmap[wnr], ifnmap[lsr]] += n
		tot += 1

# oh shit, this may have been due to the fact that sparse matrices are saved incorrectly!
tot = 0
targ = '/Users/davidlea/Desktop/testing/task_data/combined_win_data'
with open(targ, 'w') as f:
	for wnr, v in win_tab.iteritems():
		for lsr, n in v.iteritems():
			if not tot % 1000:
				print tot
			f.write('%i,%i,%i\n' % (ifnmap[wnr], ifnmap[lsr], n))
			tot += 1

targ = '/Users/davidlea/Desktop/testing/task_data/combined_win_data_fullids'
tot = 0
with open(targ, 'w') as f:
	for wnr, v in win_tab.iteritems():
		for lsr, n in v.iteritems():
			if not tot % 1000:
				print tot
			f.write('%s,%s,%i\n' % (wnr, lsr, n))
			tot += 1

# targ = '/Users/davidlea/Desktop/testing/task_data/correct_win_matrix.mtx'
# io.mmwrite(targ, win_matrix.astype(int))