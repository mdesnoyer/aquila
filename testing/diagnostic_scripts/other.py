"""
Uses aquila_train to train a model from the configuration specified in
config.py.

NOTES:
    Either my creation of the win matrices was buggy, or we have an issue
    with the lil matrices used to create the sparse matrices. In practice,
    it appears to be both. From now on, we won't be using sparse matrices,
    we're going to be reading an enumeration of all win events in the data.
"""

from PIL import Image
import numpy as np
from training.input import InputManager
from training.input import get_enqueue_op
from aquila_train import _tower_loss
from net import aquila_model as aquila
from config import *
import tensorflow as tf
from net import slim

# first, create a null image
imarray = np.random.rand(314, 314, 3) * 255
im = Image.fromarray(imarray.astype('uint8')).convert('RGBA')
im.save(NULL_IMAGE)

global_step = tf.get_variable(
                'global_step', [],
                initializer=tf.constant_initializer(0), trainable=False)

lr = 0.0
opt = tf.train.RMSPropOptimizer(lr, RMSPROP_DECAY,
                                    momentum=RMSPROP_MOMENTUM,
                                    epsilon=RMSPROP_EPSILON)
# opt = tf.train.

inputs, labels, conf, filenames = np.load('/data/testv.npy')
inputs = tf.constant(np.array([x for x in inputs]))
labels = tf.constant(np.array([x for x in labels]))
conf = tf.constant(np.array([x for x in conf]))
with tf.name_scope('testing') as scope:
    # inputs, labels, conf, filenames = im.tf_queue.dequeue_many(BATCH_SIZE)
    logits = aquila.inference(inputs, abs_feats, for_training=True,
                              restore_logits=restore_logits, scope=scope,
                              regularization_strength=WEIGHT_DECAY)

init = tf.initialize_all_variables()
sess = tf.InteractiveSession()

sess.run(init)


nacc = float("inf")
initacc = 0.5
coef = 0.99
pinputs = None
plogits = None
cnt = 0
while nacc:
  nlogits = sess.run(logits[0])
  # if pinputs is None:
  #   pinputs = ninputs
  if plogits is None:
    plogits = nlogits
  # assert np.all(pinputs == ninputs), 'Input mismatch WTFFF'
  # pinputs = ninputs
  assert np.all(plogits == nlogits), 'Logit mismatch WTFFF'
  plogits = nlogits
  print cnt
  cnt += 1
