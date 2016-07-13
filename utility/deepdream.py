# this will extract the features associated with some set of images.
#
# this is for 413d96ec042fa0f72215298983fa63455ddc505b

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime

import numpy as np
from functools import partial
import PIL.Image
from IPython.display import clear_output, Image, display, HTML
from __future__ import print_function

import numpy as np
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
                endpoints = slim.aquila.aquila(
                    inputs,
                    dropout_keep_prob=0.8,
                    num_abs_features=abs_feats,
                    is_training=for_training,
                    restore_logits=restore_logits,
                    scope=scope)
    return endpoints

pretrained_model_checkpoint_path = '/data/aquila_v2_snaps_run2_64maxacc/model.ckpt-170000'


# inputs, imfns = tf_queue.dequeue_many(BATCH_SIZE)
pinputs = tf.placeholder(tf.float32, shape = [299, 299, 3], name='input')
inputs = (pinputs - [[[92., 85., 82.]]]) / 255.
inputs = tf.expand_dims(inputs, 0)

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

# dest = '/data/deepdream/params'
# restorer.save(sess, dest)
# with open('/data/deepdream/graph.pb', 'w') as f:
#   f.write(G.as_graph_def().SerializeToString())

layers = [op.name for op in graph.get_operations() if op.type=='Conv2D']
layers = [x.replace('testtrain/testing/', '') for x in layers]

print('Number of layers', len(layers))


# Helper functions for TF Graph visualization

def visstd(a, s=0.1):
    '''Normalize the image range for visualization'''
    return (a-a.mean())/max(a.std(), 1e-4)*s + 0.5


def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add()
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = "<stripped %d bytes>"%size
    return strip_def


def rename_nodes(graph_def, rename_func):
    res_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = res_def.node.add()
        n.MergeFrom(n0)
        n.name = rename_func(n.name)
        for i, s in enumerate(n.input):
            n.input[i] = rename_func(s) if s[0]!='^' else '^'+rename_func(s[1:])
    return res_def


# Visualizing the network graph. Be sure expand the "mixed" nodes to see their
# internal structure. We are going to visualize "Conv2D" nodes.
def T(layer):
    '''Helper for getting layer output tensor'''
    return graph.get_tensor_by_name("testtrain/testing/%s:0" % layer)

feature_nums = [int(T(name).get_shape()[-1]) for name
                in layers]
print('Total number of feature channels:', sum(feature_nums))

def tffunc(*argtypes):
    '''Helper that transforms TF-graph generating function into a regular one.
    See "resize" function below.
    '''
    placeholders = list(map(tf.placeholder, argtypes))
    def wrap(f):
        out = f(*placeholders)
        def wrapper(*args, **kw):
            return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))
        return wrapper
    return wrap


# Helper function that uses TF to resize an image
def resize(img, size):
    img = tf.expand_dims(img, 0)
    return tf.image.resize_bilinear(img, size)[0,:,:,:]
resize = tffunc(np.float32, np.int32)(resize)


def calc_grad_tiled(img, t_grad, tile_size=512):
    '''Compute the value of tensor t_grad over the image in a tiled way.
    Random shifts are applied to the image to blur tile boundaries over
    multiple iterations.'''
    sz = tile_size
    h, w = img.shape[:2]
    sx, sy = np.random.randint(sz, size=2)
    img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
    grad = np.zeros_like(img)
    for y in range(0, max(h-sz//2, sz),sz):
        for x in range(0, max(w-sz//2, sz),sz):
            sub = img_shift[y:y+sz,x:x+sz]
            g = sess.run(t_grad, {pinputs:sub})
            grad[y:y+sz,x:x+sz] = g
    return np.roll(np.roll(grad, -sx, 1), -sy, 0)

k = np.float32([1,4,6,4,1])
k = np.outer(k, k)
k5x5 = k[:,:,None,None]/k.sum()*np.eye(3, dtype=np.float32)


def lap_split(img):
    '''Split the image into lo and hi frequency components'''
    with tf.name_scope('split'):
        lo = tf.nn.conv2d(img, k5x5, [1,2,2,1], 'SAME')
        lo2 = tf.nn.conv2d_transpose(lo, k5x5*4, tf.shape(img), [1,2,2,1])
        hi = img-lo2
    return lo, hi


def lap_split_n(img, n):
    '''Build Laplacian pyramid with n splits'''
    levels = []
    for i in range(n):
        img, hi = lap_split(img)
        levels.append(hi)
    levels.append(img)
    return levels[::-1]


def lap_merge(levels):
    '''Merge Laplacian pyramid'''
    img = levels[0]
    for hi in levels[1:]:
        with tf.name_scope('merge'):
            img = tf.nn.conv2d_transpose(img, k5x5*4, tf.shape(hi), [1,2,2,1]) + hi
    return img


def normalize_std(img, eps=1e-10):
    '''Normalize image by making its standard deviation = 1.0'''
    with tf.name_scope('normalize'):
        std = tf.sqrt(tf.reduce_mean(tf.square(img)))
        return img/tf.maximum(std, eps)


def lap_normalize(img, scale_n=4):
    '''Perform the Laplacian pyramid normalization.'''
    img = tf.expand_dims(img,0)
    tlevels = lap_split_n(img, scale_n)
    tlevels = list(map(normalize_std, tlevels))
    out = lap_merge(tlevels)
    return out[0,:,:,:]


def render_lapnorm(t_obj, img0, iter_n=10, step=1.0, octave_n=3,
                   octave_scale=1.4, lap_n=4):
    t_score = tf.reduce_mean(t_obj) # defining the optimization objective
    t_grad = tf.gradients(t_score, pinputs)[0] # behold the power of automatic
    # differentiation!
    # build the laplacian normalization graph
    lap_norm_func = tffunc(np.float32)(partial(lap_normalize, scale_n=lap_n))

    img = img0.copy()
    for octave in range(octave_n):
        if octave>0:
            hw = np.float32(img.shape[:2])*octave_scale
            img = resize(img, np.int32(hw))
        for i in range(iter_n):
            g = calc_grad_tiled(img, t_grad)
            g = lap_norm_func(g)
            img += g*step
            print('.', end = ' ')
    return img

def dream(layer, channel):
    t_obj = T(layer)[:, :, :, channel]
    img0 = (np.random.uniform(size=(299,299,3)) * 255).astype(np.float32)
    return render_lapnorm(t_obj, img0)

# img_noise = (np.random.uniform(size=(299,299,3)) * 255).astype(np.uint8)
layer = 'mixed_8x8x2048b/branch_pool/Conv/Conv2D'
channel = 104
img = dream(layer, channel)