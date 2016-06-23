# assemble the training set.
# 0 = <= 29
# 1 = 30 - 39
# 2 = 40 - 49
# 3 = 50+
# 4 = Unk

# male = 0
# female = 5

# 0 = male / <= 29
# 1 = male / 30 - 39
# 2 = male / 40 - 49
# 3 = male / 50+
# 4 = female / <= 29
# 5 = female / 30 - 39
# 6 = female / 40 - 49
# 7 = female / 50+
# 8 = Unk / Unk

# bins = Counter()
# for i in f:
#     a, b, c, d, e = i.split(',')
#     bins[d] += 1

# for i in f:
#     a, b, c, d, e = i.split(',')
#     bins[c] += 1

# i.e.,
# (a, b) = < # of times a beat b by demo> < # of times b beat a by demo >

from collections import defaultdict as ddict
import locale
import sys

try:
    # for linux
    locale.setlocale(locale.LC_ALL, 'en_US.utf8')
except:
    # for mac
    locale.setlocale(locale.LC_ALL, 'en_US')

test_size = 1000  # the number of EDGES in the testing set, such that at
# least one image in each pair (i.e., edge) is in the testing set. What we
# could do is greedily add images, by selecting the images with the minimum
# number of edges.

# first step is to determine all the available images that we've got
all_ims = dict()
with open('/data/aquila_data/avail_images/imagelist', 'r') as f:
    for line in f:
        img = line.strip()
        all_ims[img.split('.')[0]] = img

age_bin_correspondence = {  # age bin 2 age interval
    0: (0, 19),
    1: (20, 29),
    2: (30, 39),
    3: (40, 49),
    4: (50, float("inf")),
    5: 'unknown'
}

gend_bin_correspondence = {  # gender bin to gender
    0: 'male',
    1: 'female',
    2: 'unknown'
}

age_gend_abs_bin = {  # age bin, gender bin to absolute bin
    (0, 0): 0,
    (1, 0): 1,
    (2, 0): 2,
    (3, 0): 3,
    (4, 0): 4,
    (0, 1): 5,
    (1, 1): 6,
    (2, 1): 7,
    (3, 1): 8,
    (4, 1): 9,
    # 10 is unknown
}

old_age_bins = {
    '18-29': 0,
    '30-39': 2,
    '40-49': 3,
    '20-29': 1,
    '50-59': 4,
    '': 5,
    '60-69': 4,
    'None': 5,
    '0-19': 0,
    '70+': 4}

new_age_bins = {(0, 19): 0,
            (20, 29): 1,
            (30, 39): 2,
            (40, 49): 3,
            (50, 59): 4,
            (60, 69): 4,
            (70, 100): 4,
            (-7977, -7977): 1}


def new_age_to_bin(nage):
    """converts a new age to a bin"""
    if not nage:
        return 6
    else:
        k = int(nage)
        for a, b in new_age_bins:
            if a <= k and k <= b:
                return new_age_bins[(a, b)]


def old_age_to_bin(oage):
    """converts an old age to a bin"""
    return old_age_bins[oage]


def gender_to_bin(gend):
    """ works for both """
    if gend == 'male':
        return 0
    elif gend == 'female':
        return 1
    else:
        return 2


def age_gend_bin_to_comb(age_bin, gend_bin):
    """ combines an age bin and a gender bin """
    return age_gend_abs_bin.get((age_bin, gend_bin), 10)


import locale

try:
    # for linux
    locale.setlocale(locale.LC_ALL, 'en_US.utf8')
except:
    # for mac
    locale.setlocale(locale.LC_ALL, 'en_US')


data = ddict(lambda: [0] * 22)
data_test = ddict(lambda: [0] * 22)

tot = 0
bad = 0
bad_ims_v1 = set()
bad_ims_v2 = set()
good_im_cnt = 0

with open('/data/aquila_data/mturk_task_v1/data', 'r') as f:
    for line in f:
        win, lose, gender, age_b, wid = line.strip().split(',')
        win = win.split('.')[0]
        lose = lose.split('.')[0]
        if win not in all_ims:
            bad_ims_v1.add(win)
            continue
        if lose not in all_ims:
            bad_ims_v1.add(lose)
            continue
        good_im_cnt += 2
        if not win or not lose:
            # some records have only a win or a lose, not both. arg.
            bad += 1
            continue
        if win < lose:
            key = (win, lose)
            base_v = 0
        elif win > lose:
            key = (lose, win)
            base_v = 11  # offset the bin by 8 because it's a loss

        age_b = old_age_to_bin(age_b)
        gend_b = gender_to_bin(gender)
        demo_idx = age_gend_bin_to_comb(age_b, gend_b)

        data[key][demo_idx + base_v] += 1
        tot += 1
        if not tot % 10000:
            v2 = locale.format("%d", tot, grouping=True)
            bs = locale.format("%d", bad + len(bad_ims_v1), grouping=True)
            print '[v1] %s records collected (%s bad)' % (v2, bs)
tot_old = len(data)

with open('/data/aquila_data/mturk_task_v2/data', 'r') as f:
    for line in f:
        win, lose, gender, age, wid, tid = line.strip().split(',')
        if win.split('.')[0] not in all_ims:
            bad_ims_v2.add(win)
            continue
        if lose.split('.')[0] not in all_ims:
            bad_ims_v2.add(lose)
            continue
        good_im_cnt += 2
        if not win or not lose:
            bad += 1
            continue
        if win < lose:
            key = (win, lose)
            base_v = 0
        elif win > lose:
            key = (lose, win)
            base_v = 11  # offset the bin by 8 because it's a loss
        age_b = new_age_to_bin(age)
        gend_b = gender_to_bin(gender)
        demo_idx = age_gend_bin_to_comb(age_b, gend_b)
        data[key][demo_idx + base_v] += 1
        tot += 1
        if not tot % 10000:
            v2 = locale.format("%d", tot, grouping=True)
            bs = locale.format("%d", bad + len(bad_ims_v2), grouping=True)
            print '[v2] %s records collected (%s bad)' % (v2, bs)


with open('/data/aquila_data/mturk_task_v2/data_test', 'r') as f:
    for line in f:
        win, lose, gender, age, wid, tid = line.strip().split(',')
        if win.split('.')[0] not in all_ims:
            bad_ims_v2.add(win)
            continue
        if lose.split('.')[0] not in all_ims:
            bad_ims_v2.add(lose)
            continue
        good_im_cnt += 2
        if not win or not lose:
            bad += 1
            continue
        if win < lose:
            key = (win, lose)
            base_v = 0
        elif win > lose:
            key = (lose, win)
            base_v = 11  # offset the bin by 8 because it's a loss
        age_b = new_age_to_bin(age)
        gend_b = gender_to_bin(gender)
        demo_idx = age_gend_bin_to_comb(age_b, gend_b)
        data_test[key][demo_idx + base_v] += 1
        tot += 1
        if not tot % 10000:
            v2 = locale.format("%d", tot, grouping=True)
            bs = locale.format("%d", bad + len(bad_ims_v2), grouping=True)
            print '[v2 test] %s records collected (%s bad)' % (v2, bs)


tot_new = len(data) - tot_old


with open('/data/aquila_data/combined', 'w') as f:
    for n, k in enumerate(sorted(data.keys())):
        if not n % 10000:
            v2 = locale.format("%d", n, grouping=True)
            print '%s records written' % v2
        a, b = k
        winstr = ','.join([str(x) for x in data[k]])
        f.write('%s,%s,%s\n' % (all_ims[a], all_ims[b], winstr))


with open('/data/aquila_data/combined_testing', 'w') as f:
    for n, k in enumerate(sorted(data_test.keys())):
        if not n % 10000:
            v2 = locale.format("%d", n, grouping=True)
            print '%s records written' % v2
        a, b = k
        winstr = ','.join([str(x) for x in data_test[k]])
        f.write('%s,%s,%s\n' % (all_ims[a], all_ims[b], winstr))


with open('/data/aquila_data/bad_v1', 'w') as f:
    f.write('\n'.join(bad_ims_v1))

with open('/data/aquila_data/bad_v2', 'w') as f:
    f.write('\n'.join(bad_ims_v2))

v1o = locale.format("%d", tot_old, grouping=True)
v2o = locale.format("%d", tot_new, grouping=True)

v2 = locale.format("%d", n, grouping=True)
print '%s obtained from v1, %s obtained from v2' % (v1o, v2o)
print '%s records written' % v2
print 'All files written'