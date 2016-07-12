"""
A function to generate a heatmap using Aquila
"""
# this will extract the features associated with some set of images.
#
# this is for 413d96ec042fa0f72215298983fa63455ddc505b

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime

import numpy as np
from functools import partial
from PIL import Image

import tensorflow as tf

from net.slim import slim
from config import *

import locale

try:
    # for linux
    locale.setlocale(locale.LC_ALL, 'en_US.utf8')
except:
    # for mac
    locale.setlocale(locale.LC_ALL, 'en_US')

def fmt_num(num):
    '''
    accepts a number and then formats it for the locale.
    '''
    return locale.format("%d", num, grouping=True)

BATCHNORM_MOVING_AVERAGE_DECAY = 0.9997
MOVING_AVERAGE_DECAY = 0.9999


def inference(inputs, abs_feats=1024, for_training=True,
              restore_logits=True, scope=None,
              regularization_strength=0.000005):
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
            with slim.arg_scope([slim.variables.variable], device='/cpu:0'):
                logits, endpoints = slim.aquila.aquila(
                    inputs,
                    dropout_keep_prob=0.8,
                    num_abs_features=abs_feats,
                    is_training=for_training,
                    restore_logits=restore_logits,
                    scope=scope)
    return logits, endpoints

pretrained_model_checkpoint_path = '/data/aquila_v2_snaps/model.ckpt-150000'
# pretrained_model_checkpoint_path = '/data/aquila_models/model.ckpt-150000'


def get_ph():
    """
    returns a placeholder for an image.
    """
    return tf.placeholder(tf.float32, shape=[299, 299, 3])


pinputs = [get_ph() for x in range(BATCH_SIZE)]
msinputs = [(x - [[[92., 85., 82.]]]) / 255. for x in pinputs]
inputs = tf.pack(msinputs, name='inputs')

with tf.variable_scope('testtrain') as varscope:
    logits, endpoints = inference(inputs, abs_feats, for_training=False,
                                  restore_logits=restore_logits,
                                  scope='testing',
                                  regularization_strength=WEIGHT_DECAY)

graph = tf.get_default_graph()

init = tf.initialize_all_variables()
sess = tf.InteractiveSession()
sess.run(init)

variables_to_restore = tf.get_collection(
                slim.variables.VARIABLES_TO_RESTORE)
restorer = tf.train.Saver(variables_to_restore)

restorer.restore(sess, pretrained_model_checkpoint_path)
print('%s: Pre-trained model restored from %s' %
                    (datetime.now(), pretrained_model_checkpoint_path))

dh = 314
dw = 558

targ_img = Image.open('/data/targ.jpg')

# the max expansion factor for the target region
max_fact = min(targ_img.size[1] / float(dh), targ_img.size[0] / float(dw))
# the min reduction factor for the target region
min_fact = 0.1


def get_random_crop(targ_img):
    # gets a random crop
    #
    # returns the cropped image, [top, bottom, left, right] coordinates.
    fact = np.random.rand() * (max_fact - min_fact) + min_fact
    ch = int(fact * dh)
    cw = int(fact * dw)
    hc1 = np.random.randint(targ_img.size[1] - ch)
    wc1 = np.random.randint(targ_img.size[0] - cw)
    hc2 = hc1 + ch
    wc2 = wc1 + cw
    cim = np.array(targ_img.crop([wc1, hc1, wc2, hc2]).resize((299, 299)))
    return cim, [hc1, hc2, wc1, wc2]

def populate_ph(targ_img, phds):
    """
    Produces a feed dict for the data.
    """
    fd = dict()
    all_c = []
    for phd in phds:
        img, coords = get_random_crop(targ_img)
        fd[phd] = img
        all_c.append(coords)
    return fd, all_c

sum_samples = np.zeros((targ_img.size[1], targ_img.size[0]))
min_map = np.ones_like(sum_samples) * float("inf")
max_map = np.ones_like(sum_samples) * -float("inf")
count_samples = np.zeros((targ_img.size[1], targ_img.size[0]))

tot_samps = 0
min_val = float("inf")
max_val = -float("inf")

while tot_samps < 20000:
    fd, all_c = populate_ph(targ_img, pinputs)
    logits_n = sess.run(logits, feed_dict=fd)

    for n in range(len(logits_n)):
        mean_log = np.mean(logits_n[n])
        hc1, hc2, wc1, wc2 = all_c[n]
        sum_samples[hc1:hc2, wc1:wc2] += mean_log
        min_map[hc1:hc2, wc1:wc2] = np.minimum(min_map[hc1:hc2, wc1:wc2],
                                               mean_log)
        max_map[hc1:hc2, wc1:wc2] = np.maximum(max_map[hc1:hc2, wc1:wc2],
                                               mean_log)
        count_samples[hc1:hc2, wc1:wc2] += 1

    tot_samps += len(logits_n)
    min_val = min(min_val, np.min(logits_n))
    max_val = max(max_val, np.max(logits_n))
    print('Samples:', tot_samps)

datas = [min_map, max_map, sum_samples, count_samples, max_val, min_val]
np.save('/tmp/heatmap', datas)

min_map, max_map, sum_samples, count_samples, max_val, min_val = np.load(
    '/data/aquila_extractions/heatmap.npy')
min_map[min_map==float("inf")] = min_val
max_map[max_map==-float("inf")] = min_val

mean_map = sum_samples / np.maximum(count_samples, 1)

def cmap(arr, mapname='autumn', vmin=None, vmax=None):
    if vmin is None:
        vmin = np.min(arr)
    if vmax is None:
        vmax = np.max(arr)
    nrm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.ScalarMappable(nrm, mapname)
    return (np.array(cmap.to_rgba(arr)) * 255).astype(np.uint8)[:, :, :3]

from PIL import Image
targ_img = Image.open('/data/targ.jpg')
overlay = Image.fromarray(cmap(np.exp(mean_map)))
blend = Image.blend(targ_img, overlay, 0.8)