"""
Uses aquila_train to train a model from the configuration specified in 
config.py.
"""

from training.input import InputManager
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

fnmap = dict()
print 'Loading index to filename map'
with open(FILE_MAP_LOC, 'r') as f:
    for line in f:
        idx, fn = line.strip().split(',')
        fnmap[int(idx)] = fn + '.jpg'
print 'Loading win matrix'
win_matrix = sparse.lil_matrix(io.mmread(WIN_MATRIX_LOC).astype(np.uint8))

outQ = tf.FIFOQueue(BATCH_SIZE*16, [tf.float32, tf.float32], shapes=[[299, 299,
                                                                    3],
                                                           [BATCH_SIZE]])
fn_phds = [tf.placeholder(tf.string, shape=[]) for _ in range(BATCH_SIZE)]
lab_phds = [tf.placeholder(tf.int32, 
                           shape=[BATCH_SIZE]) for _ in range(BATCH_SIZE)]
enq_op = get_enqueue_op(fn_phds, lab_phds, outQ)

imgr = InputManager(win_matrix, fnmap, IMG_DIR, outQ, fn_phds, lab_phds,
                    enq_op, BATCH_SIZE, num_epochs=NUM_EPOCHS, num_threads=1)

aquila_train.train(imgr, imgr.num_ex_per_epoch)


