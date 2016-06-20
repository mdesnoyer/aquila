from config import *
from PIL import Image
import numpy as np
import tensorflow as tf
from training.input import InputManager
from training.input import get_enqueue_op
from net import aquila_model as aquila
from net.slim import slim

# gtd = dict()  # the ground truth data
# with open(TRAIN_DATA, 'r') as f:
#     for line in f:
#         a = line.split(',')[0]
#         b = line.split(',')[1]
#         dat = [int(x) for x in line.split(',')[2:]]
#         gtd[(a, b)] = dat[:DEMOGRAPHIC_GROUPS]
#         gtd[(b, a)] = dat[DEMOGRAPHIC_GROUPS:]

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
                  enqueue_op, num_epochs=2, num_qworkers=1)

ims, wins, conf, fns = tf_queue.dequeue_many(23)
labels = aquila.inference(ims, abs_feats, for_training=True,
                          restore_logits=restore_logits, scope='',
                          regularization_strength=WEIGHT_DECAY)[0]
accuracy = aquila.accuracy(labels, wins)
op = [accuracy, labels, ims, wins, conf, fns]

init = tf.initialize_all_variables()
sess = tf.InteractiveSession()
sess.run(init)
im.start(sess)

