"""
Uses aquila_train to train a model from the configuration specified in 
config.py.

NOTES:
    Either my creation of the win matrices was buggy, or we have an issue
    with the lil matrices used to create the sparse matrices. In practice,
    it appears to be both. From now on, we won't be using sparse matrices,
    we're going to be reading an enumeration of all win events in the data.
"""

from training.input import InputManagerWinList
from training.input import get_enqueue_op
import aquila_train
import config

import numpy as np
import tensorflow as tf

from scipy import io
from scipy import sparse

IMG_DIR = '/data/images'
FILE_MAP_LOC = '/data/datasets/idx_2_id'
BATCH_SIZE = config.BATCH_SIZE
NUM_EPOCHS = config.NUM_EPOCHS

if config.subset == 'train':
    WIN_MATRIX_LOC = '/data/datasets/train/win_matrix.mtx'
else:
    WIN_MATRIX_LOC = '/data/datasets/test/win_matrix.mtx'

WIN_LIST_LOC = '/data/datasets/combined_win_data'

fnmap = dict()
print 'Loading index to filename map'
with open(FILE_MAP_LOC, 'r') as f:
    for line in f:
        idx, fn = line.strip().split(',')
        fnmap[int(idx)] = fn + '.jpg'
print 'Constructing win list'
win_list = []
with open(WIN_LIST_LOC, 'r') as f:
    for line in f:
        a, b, n = line.split(',')
        win_list.append([int(a), int(b), int(n)])

# win_matrix = sparse.lil_matrix(io.mmread(WIN_MATRIX_LOC).astype(np.uint8))

outQ = tf.FIFOQueue(BATCH_SIZE*16, [tf.float32, tf.float32], shapes=[[299, 299,
                                                                    3],
                                                           [BATCH_SIZE]])
fn_phds = [tf.placeholder(tf.string, shape=[]) for _ in range(BATCH_SIZE)]
lab_phds = [tf.placeholder(tf.int32, 
                           shape=[BATCH_SIZE]) for _ in range(BATCH_SIZE)]
enq_op = get_enqueue_op(fn_phds, lab_phds, outQ)

# imgr = InputManager(win_matrix, fnmap, IMG_DIR, outQ, fn_phds, lab_phds,
#                     enq_op, BATCH_SIZE, num_epochs=NUM_EPOCHS, num_threads=1,
#                     debug_dir='/data/training_epoch_sequence',
#                     single_win_mapping=True)

imgr = InputManagerWinList(win_list, fnmap, IMG_DIR, outQ, fn_phds, lab_phds,
                    enq_op, BATCH_SIZE, num_epochs=NUM_EPOCHS, num_threads=1,
                    debug_dir='/data/training_epoch_sequence',
                    single_win_mapping=True)

sess = tf.InteractiveSession()
imgr.start(sess)
# aquila_train.train(imgr, imgr.num_ex_per_epoch)


