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

sess = tf.InteractiveSession()