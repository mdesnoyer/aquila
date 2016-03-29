"""
producing the training data for the tensorflow task
"""

# legacy is of the form [a, b, N a beats b, N b beats a]
from collections import Counter
from collections import defaultdict as ddict
import numpy as np
from glob import glob
import os
# # let's just make sure that everything in column 1 is duplicated in column 2
# wins = Counter()
# losses = Counter()
# with open('legacy.csv', 'r') as f:
# 	for line in f:
# 		a, b, c, d = line.split(',')
# 		wins[(a, b)] = int(c)
# 		losses[(a, b)] = int(d)
# 		if int(c) and int(d):
# 			test_pair = (a,b)

# for a, b in wins.keys():
# 	if wins[(b, a)] != losses[(a, b)]:
# 		print 'error!'
# 		break

# # that is indeed the case!
im_src = '/data/aquila_training_images'
dat_dst = '/repos/aquila/task_data/aggregated_data/'
dat_src = '/repos/aquila/task_data/win_lists'

wins = Counter()  # pairwise wins
tot_wins = Counter()  # total wins for an image
tot_losses = Counter()  # total loses for an image

allowable_ims = glob(os.path.join(im_src, '*.jpg'))
allowable_ims = set([x.split('/')[-1].split('.')[0] for x in allowable_ims])

with open(os.path.join(dat_src, 'legacy.csv'), 'r') as f:
    print 'reading legacy data'
    for line in f:
        a, b, c, d = line.split(',')
        a = a.split('.')[0]
        b = b.split('.')[0]
        if a not in allowable_ims:
            continue
        if b not in allowable_ims:
            continue
        wins[(a, b)] = int(c)
        tot_wins[a] += int(c)
        tot_losses[b] += int(c)

with open(os.path.join(dat_src, 'newtask.csv'), 'r') as f:
    print 'reading new data'
    for line in f:
        a, b, c = line.split(',')
        wins[(a, b)] += int(c)
        tot_wins[a] += int(c)
        tot_losses[b] += int(c)

outcomes = ddict(lambda: [0, 0])
print 'creating outcomes dict'
for (a, b), c in wins.iteritems():
    if a > b:
        outcomes[(a, b)][0] += c
    elif b > a:
        outcomes[(a, b)][1] += c
    else:
        raise Exception('Gah! Two equivalent keys!')

#  the final output data CSV will be:
#  (a, b, N a>b, N b>a, prior_a>b, prior_b>a)
#
# where the priors are given by:
#
# (wij + g_i / (g_i + g_j)) / (wij + wji + 1)
id_2_idx = dict()
idx_2_id = dict()

imgs = list(set(tot_wins.keys() + tot_losses.keys()))
for n, cid in enumerate(imgs):
    id_2_idx[cid] = n
    idx_2_id[n] = cid


def get_prior(a, b):
    """
    Returns the priors p_a>b, p_b>a, assuming a and b are served in
    lexicographic order

    Args:
        a: An image
        b: An image

    Returns: The priors for the images.
    """
    w_ab, w_ba = outcomes[(a,b)]
    g_a = float(tot_wins[a]) / (tot_wins[a] + tot_losses[a])
    g_b = float(tot_wins[b]) / (tot_wins[b] + tot_losses[b])
    p_ab = (w_ab + (g_a / (g_a + g_b))) / (w_ab + w_ba + 1)
    p_ba = (w_ba + (g_b / (g_a + g_b))) / (w_ab + w_ba + 1)
    return p_ab, p_ba

all_d = []
for n, ((a, b), (c, d)) in enumerate(outcomes.iteritems()):
    print n
    p_ab, p_ba = get_prior(a,b)
    a = id_2_idx[a]
    b = id_2_idx[b]
    all_d.append([a, b, c, d, p_ab, p_ba])

np.save(os.path.join(dat_dst, 'win_data'), np.array(all_d))

with open(os.path.join(dat_dst, 'idx_2_id'), 'w') as f:
    for k, v in idx_2_id.iteritems():
        f.write('%i,%s\n' % (k, v))







