"""
This script uses test data in /tmp/aquila_test_data to test the input manager.
"""
from config import *
from PIL import Image
import cv2
import numpy as np
import tensorflow as tf
from training.input import InputManager
from training.input import get_enqueue_op

gtd = dict()  # the ground truth data
with open('/tmp/aquila_test_data/combined', 'r') as f:
    for line in f:
        a = line.split(',')[0]
        b = line.split(',')[1]
        dat = [int(x) for x in line.split(',')[2:]]
        gtd[(a, b)] = dat[:DEMOGRAPHIC_GROUPS]
        gtd[(b, a)] = dat[DEMOGRAPHIC_GROUPS:]

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
                  enqueue_op, num_epochs=2, num_qworkers=1)

sess = tf.InteractiveSession()

im.start(sess)
while not im.should_stop():
    print 'Attempting to dequeue'
    a, b, c, d = sess.run(tf_queue.dequeue_many(22))
    print 'Dequeue successful'
im.stop()

x = a[0].astype(np.uint8)
fn = d[0]
targ = np.array(Image.open(fn))
cv2.imshow('tensorflow', x[:, :, ::-1])
cv2.imshow('target', targ[:, :, ::-1])

# okay so it appears the images work.
keys = [xv.split('/')[-1] for xv in d]
for i in range(len(keys)):
    for j in range(len(keys)):
        assert np.all(b[i, j] == gtd.get((keys[i], keys[j]), [0] *
                                  DEMOGRAPHIC_GROUPS)), 'Doesnt match for %i, ' \
                                                       '%i' %(i, j)

# another problem with batch gen
from training.input import *
pairs, labels = read_in_data(DATA_SOURCE)
bg = batch_gen(pairs)
next_batch = bg.next()
batch, wm = gen_labels(next_batch, labels)