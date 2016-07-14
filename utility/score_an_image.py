# score CNN

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image

import copy
from datetime import datetime
import os.path
import re
import time
import sys

import numpy as np
import tensorflow as tf

from net import aquila_model as aquila
from net.slim import slim
from config import *

MEAN_CHANNEL_VALS = [[[92.366, 85.133, 81.674]]]
MEAN_CHANNEL_VALS = np.array(MEAN_CHANNEL_VALS).round().astype(np.float32)

BATCHNORM_MOVING_AVERAGE_DECAY = 0.9997
MOVING_AVERAGE_DECAY = 0.9999

def inference(inputs, abs_feats=1024, for_training=True,
              restore_logits=True, scope=None,
              regularization_strength=0.000005):
    """
    Exports an inference op, along with the logits required for loss
    computation.
    :param inputs: An N x 299 x 299 x 3 sized float32 tensor (images)
    :param abs_feats: The number of abstract features to learn.
    :param for_training: Boolean, whether or not training is being performed.
    :param restore_logits: Restore the logits. This should only be done if the
    model is being trained on a previous snapshot of Aquila. If training from
    scratch, or transfer learning from inception, this should be false as the
    number of abstract features will likely change.
    :param scope: The name of the tower (i.e., GPU) this is being done on.
    :return: Logits, aux logits.
    """
    # Parameters for BatchNorm.
    batch_norm_params = {
        # Decay for the moving averages.
        'decay': BATCHNORM_MOVING_AVERAGE_DECAY,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
    }

    # Set weight_decay for weights in Conv and FC layers.
    with slim.arg_scope([slim.ops.conv2d, slim.ops.fc],
            weight_decay=regularization_strength):
        with slim.arg_scope([slim.ops.conv2d],
                stddev=0.1,
                activation=tf.nn.relu,
                batch_norm_params=batch_norm_params):
            # i'm disabling batch normalization, because I'm concerned that
            # even though the authors claim it preserves representational
            # power, I don't believe their claim and I'm concerned about the
            # distortion it may introduce into the images.
            # Force all Variables to reside on the CPU.
            with slim.arg_scope([slim.variables.variable], device='/cpu:0'):
                logits, endpoints = slim.aquila.aquila(
                    inputs,
                    dropout_keep_prob=0.8,
                    num_abs_features=abs_feats,
                    is_training=for_training,
                    restore_logits=restore_logits,
                    scope=scope)
    return logits, endpoints

inputs = tf.placeholder(tf.float32, shape=[1, 299, 299, 3])

with tf.variable_scope('testtrain') as varscope:
    logits, endpoints = inference(inputs, abs_feats, for_training=False,
                                  restore_logits=restore_logits,
                                  scope='testing',
                                  regularization_strength=WEIGHT_DECAY)

init = tf.initialize_all_variables()
sess = tf.InteractiveSession()
sess.run(init)
pretrained_model_checkpoint_path = '/data/aquila_v2_snaps/model.ckpt-150000'

variables_to_restore = tf.get_collection(
                slim.variables.VARIABLES_TO_RESTORE)
restorer = tf.train.Saver(variables_to_restore)
restorer.restore(sess, pretrained_model_checkpoint_path)
print('%s: Pre-trained model restored from %s' %
                    (datetime.now(), pretrained_model_checkpoint_path))

def score_im(imfn, op):
    img = Image.open(imfn)
    assert img.size[0] == 299, 'Image must be 299 x 299 x 3'
    assert img.size[1] == 299, 'Image must be 299 x 299 x 3'
    x = np.array(img)
    x = (x - MEAN_CHANNEL_VALS) / 256.
    return sess.run(op, feed_dict={inputs: x[None, :, :, :]})

im1 = '/tmp/test/old.jpeg'
im2 = '/tmp/test/new.jpeg'
op = logits
