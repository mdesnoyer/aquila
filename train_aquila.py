"""
Runs the training of Aquila.
"""

from training.input import InputManager
from training.input import get_enqueue_op
from net.slim import aquila_model as aquila
from net.slim.losses import ranknet_loss
from net.slim.losses import accuracy
# Collapse tf-slim into a single namespace.
from net.slim import ops
from net.slim import scopes
from net.slim import losses
from net.slim import variables
from net.slim.scopes import arg_scope
import tensorflow as tf
from scipy import io
from scipy import sparse
import numpy as np


# Configuration
IMG_DIR = '/other/testing_ims' # TODO: Change from testing!
FILE_MAP_LOC = '/repos/aquila/task_data/aggregated_data/idx_2_id'
WIN_MATRIX_LOC = '/repos/aquila/task_data/datasets/test/win_matrix.mtx' # TODO: Change from testing!
BATCH_SIZE = 32
BATCHNORM_MOVING_AVERAGE_DECAY = 0
NUM_ABS_FEATURES = 1024  # the number of abstract features to use
IS_TRAINING = True
RESTORE_LOGITS = True  # you should save and restore the logits

fnmap = dict()
print 'Loading index to filename map'
with open(FILE_MAP_LOC, 'r') as f:
    for line in f:
        idx, fn = line.strip().split(',')
        fnmap[int(idx)] = fn + '.jpg'
print 'Loading win matrix'
win_matrix = sparse.lil_matrix(io.mmread(WIN_MATRIX_LOC).astype(np.uint8))
# ---------  FOR TESTING  ---------- #
outQ = tf.FIFOQueue(128, [tf.float32, tf.float32], shapes=[[299, 299, 3],
                                                           [32]])
fn_phds = [tf.placeholder(tf.string, shape=[]) for _ in range(BATCH_SIZE)]
lab_phds = [tf.placeholder(tf.int32, shape=[BATCH_SIZE]) for _ in range(BATCH_SIZE)]
enq_op = get_enqueue_op(fn_phds, lab_phds, outQ)
image_inputs, labels = outQ.dequeue_many(BATCH_SIZE)
batch_norm_params = {
  # Decay for the batch_norm moving averages.
  'decay': BATCHNORM_MOVING_AVERAGE_DECAY,
  # epsilon to prevent 0s in variance.
  'epsilon': 0.001,
}

with arg_scope([ops.conv2d, ops.fc], weight_decay=0.00004):
    with arg_scope([ops.conv2d],
                        stddev=0.1,
                        activation=tf.nn.relu,
                        batch_norm_params=batch_norm_params):
        # Force all Variables to reside on the CPU.
        with arg_scope([variables.variable], device='/cpu:0'):
            logits, endpoints = aquila.aquila(
                image_inputs,
                dropout_keep_prob=0.8,
                num_abs_features=NUM_ABS_FEATURES,
                is_training=IS_TRAINING,
                restore_logits=RESTORE_LOGITS,
                scope='')

comb_loss = ranknet_loss(endpoints['logits'], labels) + \
            ranknet_loss(endpoints['aux_logits'], labels)

regularization_losses = tf.get_collection(losses.LOSSES_COLLECTION, '')

total_loss = comb_loss + regularization_losses

sess = tf.InteractiveSession()
imgr = InputManager(win_matrix, fnmap, IMG_DIR, outQ, fn_phds, lab_phds,
                    enq_op, sess, BATCH_SIZE, num_epochs=3, num_threads=1)
imgr.start()

# while not imgr.should_stop():
# --------- /FOR TESTING  ---------- #