"""
Uses aquila_train to train a model from the configuration specified in 
config.py.

NOTES:
    Either my creation of the win matrices was buggy, or we have an issue
    with the lil matrices used to create the sparse matrices. In practice,
    it appears to be both. From now on, we won't be using sparse matrices,
    we're going to be reading an enumeration of all win events in the data.
"""

from training.input import InputManager
from training.input import get_enqueue_op
import aquila_train
from config import *
import tensorflow as tf


tf_queue = tf.FIFOQueue(BATCH_SIZE, [tf.float32, tf.uint8, tf.float32,
                                     tf.string],
                        shapes=[[299, 299, 3],
                                [BATCH_SIZE, DEMOGRAPHIC_GROUPS],
                                [BATCH_SIZE, DEMOGRAPHIC_GROUPS],
                                []])
image_phds = [tf.placeholder(tf.string, shape=[]) for _ in range(BATCH_SIZE)]
label_phds = [tf.placeholder(tf.uint8, shape=[BATCH_SIZE, DEMOGRAPHIC_GROUPS])
              for _ in range(BATCH_SIZE)]
conf_phds = [tf.placeholder(tf.float32, shape=[BATCH_SIZE, DEMOGRAPHIC_GROUPS])
             for _ in range(BATCH_SIZE)]
enqueue_op = get_enqueue_op(image_phds, label_phds, conf_phds, tf_queue)

im = InputManager(image_phds, label_phds, conf_phds, tf_queue,
                  enqueue_op, num_epochs=NUM_EPOCHS,
                  num_qworkers=num_preprocess_threads)

aquila_train.train(im, im.num_ex_per_epoch)


