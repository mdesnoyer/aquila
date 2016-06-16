"""
There's some kind of issue with running aquila, so we're gonna test it again.

yey.
"""

from training.input import InputManager
from training.input import get_enqueue_op
from config import *
import tensorflow as tf
from net import aquila_model as aquila
from net.slim import slim

tf_queue = tf.FIFOQueue(BATCH_SIZE * num_gpus * 2,
                        [tf.float32, tf.float32, tf.float32, tf.string],
                        shapes=[[299, 299, 3],
                                [BATCH_SIZE, DEMOGRAPHIC_GROUPS],
                                [BATCH_SIZE, DEMOGRAPHIC_GROUPS],
                                []])
image_phds = [tf.placeholder(tf.string, shape=[]) for _ in range(BATCH_SIZE)]
label_phds = [tf.placeholder(tf.float32, shape=[BATCH_SIZE, DEMOGRAPHIC_GROUPS])
              for _ in range(BATCH_SIZE)]
conf_phds = [tf.placeholder(tf.float32, shape=[BATCH_SIZE, DEMOGRAPHIC_GROUPS])
             for _ in range(BATCH_SIZE)]
enqueue_op = get_enqueue_op(image_phds, label_phds, conf_phds, tf_queue)

im = InputManager(image_phds, label_phds, conf_phds, tf_queue,
                  enqueue_op, num_epochs=NUM_EPOCHS,
                  num_qworkers=num_preprocess_threads)

global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)

BATCH_SIZE *= num_gpus
num_batches_per_epoch = im.num_ex_per_epoch / BATCH_SIZE
max_steps = int(num_batches_per_epoch * NUM_EPOCHS)
decay_steps = int(num_batches_per_epoch * num_epochs_per_decay)
lr = tf.train.exponential_decay(initial_learning_rate,
                                global_step,
                                decay_steps,
                                learning_rate_decay_factor,
                                staircase=True)
# Create an optimizer that performs gradient descent.
opt = tf.train.RMSPropOptimizer(lr, RMSPROP_DECAY,
                                momentum=RMSPROP_MOMENTUM,
                                epsilon=RMSPROP_EPSILON)
split_batch_size = int(BATCH_SIZE / num_gpus)
inputs, labels, conf, filenames = im.tf_queue.dequeue_many(
                    split_batch_size)

logits = aquila.inference(inputs, abs_feats, for_training=True,
                              restore_logits=restore_logits,
                              regularization_strength=WEIGHT_DECAY)
aquila.loss(logits, labels, conf)
accuracy = aquila.accuracy(logits, labels)
losses = tf.get_collection(slim.losses.LOSSES_COLLECTION, 'test')

init = tf.initialize_all_variables()

# sess = tf.Session(config=tf.ConfigProto(
#         allow_soft_placement=True,
#         log_device_placement=log_device_placement))
sess = tf.InteractiveSession()
sess.run(init)
im.start(sess)
logits = sess.run(logits)
