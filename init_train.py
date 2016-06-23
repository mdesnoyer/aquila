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
import aquila_train
from config import *
import tensorflow as tf

# first, create a null image
imarray = np.random.rand(314, 314, 3) * 255
im = Image.fromarray(imarray.astype('uint8')).convert('RGBA')
im.save(NULL_IMAGE)

tf_queue = tf.FIFOQueue(BATCH_SIZE * num_gpus * 2,
                        [tf.float32, tf.float32, tf.float32, tf.string],
                        shapes=[[299, 299, 3],
                                [BATCH_SIZE, DEMOGRAPHIC_GROUPS-1],
                                [BATCH_SIZE, DEMOGRAPHIC_GROUPS-1],
                                []])
image_phds = [tf.placeholder(tf.string, shape=[]) for _ in range(BATCH_SIZE)]
label_phds = [tf.placeholder(tf.float32, shape=[BATCH_SIZE,
                                                DEMOGRAPHIC_GROUPS-1])
              for _ in range(BATCH_SIZE)]
conf_phds = [tf.placeholder(tf.float32, shape=[BATCH_SIZE,
                                               DEMOGRAPHIC_GROUPS-1])
             for _ in range(BATCH_SIZE)]
enqueue_op = get_enqueue_op(image_phds, label_phds, conf_phds, tf_queue)

im = InputManager(image_phds, label_phds, conf_phds, tf_queue,
                  enqueue_op, num_epochs=NUM_EPOCHS,
                  num_qworkers=num_preprocess_threads)


tf_queue_t = tf.FIFOQueue(BATCH_SIZE * num_gpus * 2,
                        [tf.float32, tf.float32, tf.float32, tf.string],
                        shapes=[[299, 299, 3],
                                [BATCH_SIZE, DEMOGRAPHIC_GROUPS-1],
                                [BATCH_SIZE, DEMOGRAPHIC_GROUPS-1],
                                []])
image_phds_t = [tf.placeholder(tf.string, shape=[]) for _ in range(BATCH_SIZE)]
label_phds_t = [tf.placeholder(tf.float32, shape=[BATCH_SIZE,
                                                 DEMOGRAPHIC_GROUPS-1])
                for _ in range(BATCH_SIZE)]
conf_phds_t = [tf.placeholder(tf.float32, shape=[BATCH_SIZE,
                                                 DEMOGRAPHIC_GROUPS-1])
               for _ in range(BATCH_SIZE)]
enqueue_op_t = get_enqueue_op(image_phds_t, label_phds_t, conf_phds_t,
                              tf_queue_t)

val_im = InputManager(image_phds_t, label_phds_t, conf_phds_t, tf_queue_t,
                      enqueue_op_t, num_epochs=999999,
                      num_qworkers=1, data_source=TEST_DATA, is_training=False)


aquila_train.train(im, val_im, im.num_ex_per_epoch)


