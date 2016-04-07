"""
Takes the legacy trial data and builds a win list of the form:

winner, loser, win count, lose count

and takes the win counts and outputs it as:

image, total wins, total losses

and outputs them as .csv files.
"""
from collections import Counter

fields = {'id': 0, 'assignment_id': 1, 'image_one': 2, 'image_two': 3,
          'image_three': 4, 'chosen_image': 5, 'created_at': 6,
          'updated_at': 7, 'condition': 8, 'trial': 9, 'worker_id': 10,
          'stimset_id': 11, 'reaction_time': 12}

win_pairs = Counter()
win_counts = Counter()
lose_counts = Counter()
with open('/data/plutonium/filtered_trials.csv', 'r') as f:
    _ = f.readline()
    for line in f:
        j = line.split(',')
        ch = j[5]
        if ch == '""':
            continue
        items = [x for x in j[2:5] if x != ch]
        # it's so inefficient to do it this way.
        if j[8] == 'KEEP':
            # then items lost to ch
            for i in items:
                win_pairs[','.join([ch, i])] += 1
                win_counts[ch] += 1
                lose_counts[i] += 1
        elif j[8] == 'RETURN':
            # then items ch lost to items
            lose_counts[ch] += 1
            for i in items:
                win_pairs[','.join([i, ch])] += 1
                win_counts[i] += 1
                lose_counts[ch] += 1
        else:
            raise Exception('Unknown trial type!')

with open('/repos/aquila/task_data/win_lists/legacy.csv', 'w') as o:
    for k, v in win_pairs.iteritems():
        w, l = k.split(',')
        n_win = str(v)
        n_lose = str(win_pairs[','.join([l, w])])
        to_w = ','.join([w, l, n_win, n_lose]) + '\n'
        o.write(to_w)

all_ims = list(set(win_counts.keys() + lose_counts.keys()))
with open('/repos/aquila/task_data/win_ratios/legacy.csv', 'w') as o:
    for im in all_ims:
        n_win = str(win_counts[im])
        n_lose = str(lose_counts[im])
        to_w = ','.join([im, n_win, n_lose]) + '\n'
        o.write(to_w)