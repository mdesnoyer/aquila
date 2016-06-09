'''
This is a test script for the more efficient version of training for aquila v2

The way it works is:
    Read in the data, then randomly select an image, adding pairs as you go to
    a set of images. Then, assemble all the data together.
'''

import numpy as np
from collections import defaultdict as ddict
import locale

try:
    # for linux
    locale.setlocale(locale.LC_ALL, 'en_US.utf8')
except:
    # for mac
    locale.setlocale(locale.LC_ALL, 'en_US')

# args
# TODO: Ensure that this are set by the conf file.
DATA_LOC = '/Users/ndufour/Desktop/mturk_data/combined'
BATCH_SIZE = 22
DEMOGRAPHIC_GROUPS = 9

assert not BATCH_SIZE % 2, 'BATCH_SIZE must be an even number'

pairs = ddict(lambda: set())
labels = dict()

# build the pairs
with open(DATA_LOC, 'r') as f:
    for n, line in enumerate(f):
        cur_data = line.strip().split(',')
        img_a = cur_data[0]
        img_b = cur_data[1]
        outcomes = [int(x) for x in cur_data[2:]]
        pairs[img_a].add(img_b)
        labels[(img_a, img_b)] = np.array(outcomes).astype(int)
        if not n % 1000:
            print 'Total read: %s' % locale.format("%d", n, grouping=True)
print 'Total read: %s' % locale.format("%d", n, grouping=True)
total_pairs = n

# there's a potential problem, in that since we're not adding an image to a
# batch more than once, we might encounter an image that was paired 20 times,
# and so the total number of images in the batch is 21. If this is the case,
# perhaps we should maintain two lists, a pending list and a current list.
# actually, no, we'll just add uneven numbers of pairs.
def yield_sets(i, pairs, pending_batches=[]):
    """
    Given pending batches, and a pair group (i -> pairs), this will construct
    and sequentially yield appropriate batches. At most two incomplete batches
    are generated.

    NOTES:
        WARNING:
            Modifies pending batches in-place.

    Args:
        i: The first pair item.
        pairs: A list of items paired with i
        pending_batches: Incomplete batches.

    Returns: An iterator over the assembled batches.
    """

    def get_next_batch():
        if len(pending_batches):
            return pending_batches.pop()
        return set()

    batch = get_next_batch()
    while pairs:
        if len(batch) == BATCH_SIZE:
            yield batch
            batch = get_next_batch()
            continue
        if len(batch | set(pairs) | set((i,))) == BATCH_SIZE - 1:
            # then it will produce aliasing.
            batch.add(i)
            while len(batch) < BATCH_SIZE - 2:
                batch.add(pairs.pop())
            yield batch
            batch = get_next_batch()
            continue
        batch.add(i)
        batch.add(pairs.pop())
        if len(batch) == BATCH_SIZE:
            yield batch
            batch = get_next_batch()
    if batch:
        yield batch
    while pending_batches:
        yield pending_batches.pop()


def batch_gen(pairs):
    cur_batch = set()
    pending_batches = []
    while True:
        np.random.shuffle(pairs)
        for i in pairs:
            pair_items = pairs[i]
            constructed_batches = yield_sets(i, pair_items, pending_batches)
            pending_batches = []
            for batch in constructed_batches:
                if len(batch) != BATCH_SIZE:
                    pending_batches.append(batch)
                else:
                    yield batch


def gen_labels(batch, labels):
    '''
    Generates a label, given a batch and the labels dictionary.

    Args:
        batch: The current batch, as a set.
        labels: The labels dictionary.

    Returns: The batch, and a win matrix, of size batch x batch x demographic
    groups
    '''
    win_matrix = np.zeros((BATCH_SIZE, BATCH_SIZE, DEMOGRAPHIC_GROUPS))
    lb = list(batch)
    for m, i in enumerate(lb):
        for n, j in enumerate(lb):
            if (i, j) in labels:
                win_matrix[m, n, :] = labels[(i,j)][:DEMOGRAPHIC_GROUPS]
                win_matrix[n, m, :] = labels[(i,j)][DEMOGRAPHIC_GROUPS:]
    return lb, win_matrix

bg = batch_gen(pairs)
next_batch = bg.next()
batch, win_matrix = gen_labels(next_batch, labels)
