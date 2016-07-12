# score CNN

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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

from glob import glob

from tensorflow.python.ops import image_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python import ops
from PIL import Image

from threading import Thread

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

# ----------------------- stuffs --------------------------------------------- #
def crop_to_bounding_box(image, offset_height, offset_width, target_height,
                         target_width, dynamic_shape=False):
  """Crops an image to a specified bounding box.
  This op cuts a rectangular part out of `image`. The top-left corner of the
  returned image is at `offset_height, offset_width` in `image`, and its
  lower-right corner is at
  `offset_height + target_height, offset_width + target_width`.
  Args:
    image: 3-D tensor with shape `[height, width, channels]`
    offset_height: Vertical coordinate of the top-left corner of the result in
                   the input.
    offset_width: Horizontal coordinate of the top-left corner of the result in
                  the input.
    target_height: Height of the result.
    target_width: Width of the result.
    dynamic_shape: Whether the input image has undertermined shape. If set to
      `True`, shape information will be retrieved at run time. Default to
      `False`.
  Returns:
    3-D tensor of image with shape `[target_height, target_width, channels]`
  Raises:
    ValueError: If the shape of `image` is incompatible with the `offset_*` or
    `target_*` arguments, and `dynamic_shape` is set to `False`.
  """
  image = ops.convert_to_tensor(image, name='image')
  _Check3DImage(image, require_static=(not dynamic_shape))
  height, width, _ = _ImageDimensions(image, dynamic_shape=dynamic_shape)

  if not dynamic_shape:
    if offset_width < 0:
      raise ValueError('offset_width must be >= 0.')
    if offset_height < 0:
      raise ValueError('offset_height must be >= 0.')

    if width < (target_width + offset_width):
      raise ValueError('width must be >= target + offset.')
    if height < (target_height + offset_height):
      raise ValueError('height must be >= target + offset.')

  cropped = array_ops.slice(image,
                            array_ops.pack([offset_height, offset_width, 0]),
                            array_ops.pack([target_height, target_width, -1]))

  return cropped


def pad_to_bounding_box(image, offset_height, offset_width, target_height,
                        target_width, dynamic_shape=False):
  """Pad `image` with zeros to the specified `height` and `width`.
  Adds `offset_height` rows of zeros on top, `offset_width` columns of
  zeros on the left, and then pads the image on the bottom and right
  with zeros until it has dimensions `target_height`, `target_width`.
  This op does nothing if `offset_*` is zero and the image already has size
  `target_height` by `target_width`.
  Args:
    image: 3-D tensor with shape `[height, width, channels]`
    offset_height: Number of rows of zeros to add on top.
    offset_width: Number of columns of zeros to add on the left.
    target_height: Height of output image.
    target_width: Width of output image.
    dynamic_shape: Whether the input image has undertermined shape. If set to
      `True`, shape information will be retrieved at run time. Default to
      `False`.
  Returns:
    3-D tensor of shape `[target_height, target_width, channels]`
  Raises:
    ValueError: If the shape of `image` is incompatible with the `offset_*` or
      `target_*` arguments, and `dynamic_shape` is set to `False`.
  """
  image = ops.convert_to_tensor(image, name='image')
  _Check3DImage(image, require_static=(not dynamic_shape))
  height, width, depth = _ImageDimensions(image, dynamic_shape=dynamic_shape)

  after_padding_width = target_width - offset_width - width
  after_padding_height = target_height - offset_height - height

  if not dynamic_shape:
    if target_width < width:
      raise ValueError('target_width must be >= width')
    if target_height < height:
      raise ValueError('target_height must be >= height')

    if after_padding_width < 0:
      raise ValueError('target_width not possible given '
                       'offset_width and image width')
    if after_padding_height < 0:
      raise ValueError('target_height not possible given '
                       'offset_height and image height')

  # Do not pad on the depth dimensions.
  if (dynamic_shape or offset_width or offset_height or
      after_padding_width or after_padding_height):
    paddings = array_ops.reshape(
      array_ops.pack([offset_height, after_padding_height,
                      offset_width, after_padding_width,
                      0, 0]),
      [3, 2])
    padded = array_ops.pad(image, paddings)
    if not dynamic_shape:
      padded.set_shape([target_height, target_width, depth])
  else:
    padded = image

  return padded

def resize_image_with_crop_or_pad(image, target_height, target_width,
                                  dynamic_shape=False):
  """Crops and/or pads an image to a target width and height.
  Resizes an image to a target width and height by either centrally
  cropping the image or padding it evenly with zeros.
  If `width` or `height` is greater than the specified `target_width` or
  `target_height` respectively, this op centrally crops along that dimension.
  If `width` or `height` is smaller than the specified `target_width` or
  `target_height` respectively, this op centrally pads with 0 along that
  dimension.
  Args:
    image: 3-D tensor of shape [height, width, channels]
    target_height: Target height.
    target_width: Target width.
    dynamic_shape: Whether the input image has undertermined shape. If set to
      `True`, shape information will be retrieved at run time. Default to
      `False`.
  Raises:
    ValueError: if `target_height` or `target_width` are zero or negative.
  Returns:
    Cropped and/or padded image of shape
    `[target_height, target_width, channels]`
  """
  image = ops.convert_to_tensor(image, name='image')
  _Check3DImage(image, require_static=(not dynamic_shape))
  original_height, original_width, _ =     _ImageDimensions(image, dynamic_shape=dynamic_shape)

  if target_width <= 0:
    raise ValueError('target_width must be > 0.')
  if target_height <= 0:
    raise ValueError('target_height must be > 0.')

  if dynamic_shape:
    max_ = math_ops.maximum
    min_ = math_ops.minimum
  else:
    max_ = max
    min_ = min

  width_diff = target_width - original_width
  offset_crop_width = max_(-width_diff // 2, 0)
  offset_pad_width = max_(width_diff // 2, 0)

  height_diff = target_height - original_height
  offset_crop_height = max_(-height_diff // 2, 0)
  offset_pad_height = max_(height_diff // 2, 0)

  # Maybe crop if needed.
  cropped = crop_to_bounding_box(image, offset_crop_height, offset_crop_width,
                                 min_(target_height, original_height),
                                 min_(target_width, original_width),
                                 dynamic_shape=dynamic_shape)

  # Maybe pad if needed.
  resized = pad_to_bounding_box(cropped, offset_pad_height, offset_pad_width,
                                target_height, target_width,
                                dynamic_shape=dynamic_shape)

  if resized.get_shape().ndims is None:
    raise ValueError('resized contains no shape.')
  if not resized.get_shape()[0].is_compatible_with(target_height):
    raise ValueError('resized height is not correct.')
  if not resized.get_shape()[1].is_compatible_with(target_width):
    raise ValueError('resized width is not correct.')
  return resized


def _ImageDimensions(images, dynamic_shape=False):
  """Returns the dimensions of an image tensor.
  Args:
    images: 4-D Tensor of shape [batch, height, width, channels]
    dynamic_shape: Whether the input image has undertermined shape. If set to
      `True`, shape information will be retrieved at run time. Default to
      `False`.
  Returns:
    list of integers [batch, height, width, channels]
  """
  # A simple abstraction to provide names for each dimension. This abstraction
  # should make it simpler to switch dimensions in the future (e.g. if we ever
  # want to switch height and width.)
  if dynamic_shape:
    return array_ops.unpack(array_ops.shape(images))
  else:
    return images.get_shape().as_list()

def _Check3DImage(image, require_static=True):
  """Assert that we are working with properly shaped image.
  Args:
    image: 3-D Tensor of shape [height, width, channels]
    require_static: If `True`, requires that all dimensions of `image` are
      known and non-zero.
  Raises:
    ValueError: if image.shape is not a [3] vector.
  """
  try:
    image_shape = image.get_shape().with_rank(3)
  except ValueError:
    raise ValueError('\'image\' must be three-dimensional.')
  if require_static and not image_shape.is_fully_defined():
    raise ValueError('\'image\' must be fully defined.')
  if any(x == 0 for x in image_shape):
    raise ValueError('all dims of \'image.shape\' must be > 0: %s' %
                     image_shape)

image_ops.resize_image_with_crop_or_pad = resize_image_with_crop_or_pad
# ----------------------- /stuffs -------------------------------------------- #
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


def get_enqueue_op(image_phds, queue):
    """
    Obtains the TensorFlow batch enqueue operation.
    :param fn_phds: Filename TensorFlow placeholders (size=[batch_size])
    :param lab_phds: Label TensorFlow placeholders (size=[batch_size])
    :param queue: The TensorFlow input queue.
    :return: The enqueue operation.
    """
    im_tensors = []
    # process the mean channel values to integers so that we can still pass
    # messages as integers
    mcv = np.array(MEAN_CHANNEL_VALS).round().astype(np.float32)
    channel_mean_tensor = tf.constant(mcv)

    for image_phd in image_phds:
        # read in the raw jpeg
        raw_im = tf.read_file(image_phd)
        # convert to jpeg
        jpeg_im = tf.image.decode_jpeg(raw_im, channels=3)
        img = tf.to_float(jpeg_im) - channel_mean_tensor
        img /= 256.
        # pad to appropriate size
        img = resize_image_with_crop_or_pad(img, 314, 558, dynamic_shape=True)
        # resize to 314 x 314
        img = tf.expand_dims(img, 0)
        img = tf.image.resize_bilinear(img, (314, 314))
        img = tf.squeeze(img, [0])
        # random crop the image
        cropped_im = tf.random_crop(img, [299, 299, 3])
        # random flip left/right
        im_tensor = tf.image.random_flip_left_right(cropped_im)
        im_tensors.append(im_tensor)
    packed_ims = tf.to_float(tf.pack(im_tensors))
    packed_fns = tf.pack(image_phds)
    enq_op = queue.enqueue_many([packed_ims, packed_fns])
    return enq_op


def qworker(enq_op, image_phds, imgs, sess):
    feed_dict = dict()
    while imgs:
        for i in range(BATCH_SIZE):
            if not imgs:
                break
            feed_dict[image_phds[i]] = imgs.pop()
        sess.run(enq_op, feed_dict=feed_dict)
    print('qworker complete')


scores = [[1,-2.2088741],[2,-1.78813],[3,-1.53112],[4,-1.35944],[5,-1.23887],
          [6,-1.14968],[7,-1.07897],[8,-1.02086],[9,-0.972179],[10,-0.930653],
          [11,-0.894573],[12,-0.86285],[13,-0.834713],[14,-0.80927174],
          [15,-0.786055],[16,-0.76463],[17,-0.744799],[18,-0.726232],
          [19,-0.708703],[20,-0.692017],[21,-0.676046],[22,-0.660703],
          [23,-0.645833],[24,-0.631345],[25,-0.617206],[26,-0.603317],
          [27,-0.58965407],[28,-0.57624],[29,-0.562964],[30,-0.54982],
          [31,-0.536794],[32,-0.523905],[33,-0.511157],[34,-0.498506],
          [35,-0.485999],[36,-0.473631],[37,-0.4614],[38,-0.449258],
          [39,-0.437356],[40,-0.425547],[41,-0.413882],[42,-0.402366],
          [43,-0.39095063],[44,-0.379681],[45,-0.368493],[46,-0.3574],
          [47,-0.346395],[48,-0.335468],[49,-0.324642],[50,-0.3138115],
          [51,-0.303033],[52,-0.292293],[53,-0.281545],[54,-0.270779],
          [55,-0.25996],[56,-0.249109],[57,-0.238192],[58,-0.22730778],
          [59,-0.216352],[60,-0.205327],[61,-0.194218],[62,-0.183042],
          [63,-0.171804],[64,-0.160432],[65,-0.148941],[66,-0.137351],
          [67,-0.12560647],[68,-0.113628],[69,-0.101433],[70,-0.0890523],
          [71,-0.076399611],[72,-0.063471],[73,-0.050288193],[74,-0.036797234],
          [75,-0.022958475],[76,-0.0087503032],[77,0.0058479787],
          [78,0.020863804],[79,0.036340461],[80,0.05225132],[81,0.0687374],
          [82,0.085806038],[83,0.103489],[84,0.121754],[85,0.140789],
          [86,0.160679],[87,0.181482],[88,0.203319],[89,0.226377],
          [90,0.250871],[91,0.277246],[92,0.306049],[93,0.338085],
          [94,0.373872],[95,0.414742],[96,0.46338164],[97,0.52417223],
          [98,0.60686182],[99,0.738631]]


def get_neon_score(valence):
    for ns, val in scores:
        if valence < val:
            return ns - 1
    return ns


pretrained_model_checkpoint_path = '/data/aquila_v2_snaps/model.ckpt-150000'
# reader = tf.train.NewCheckpointReader(pretrained_model_checkpoint_path)
# print(reader.debug_string().decode("utf-8"))

print('Constructing queue')
image_phds = [tf.placeholder(tf.string, shape=[]) for _ in range(BATCH_SIZE)]
tf_queue = tf.FIFOQueue(BATCH_SIZE * num_gpus * 2,
                        [tf.float32, tf.string],
                        shapes=[[299, 299, 3], []])

enq_op = get_enqueue_op(image_phds, tf_queue)

inputs, imfns = tf_queue.dequeue_many(BATCH_SIZE)
# for i in xrange(1):
#     with tf.device('/gpu:%d' % i):
#         with tf.name_scope('%s_%d' % (aquila.TOWER_NAME, i)) as scope:
#             with tf.variable_scope('testtrain') as varscope:
#                 logits, endpoints = inference(inputs, abs_feats, for_training=False,
#                                               restore_logits=restore_logits, scope=scope,
#                                               regularization_strength=WEIGHT_DECAY)
with tf.variable_scope('testtrain') as varscope:
    logits, endpoints = inference(inputs, abs_feats, for_training=False,
                                  restore_logits=restore_logits,
                                  scope='testing',
                                  regularization_strength=WEIGHT_DECAY)

init = tf.initialize_all_variables()
sess = tf.InteractiveSession()
sess.run(init)

variables_to_restore = tf.get_collection(
                slim.variables.VARIABLES_TO_RESTORE)
restorer = tf.train.Saver(variables_to_restore)
restorer.restore(sess, pretrained_model_checkpoint_path)
print('%s: Pre-trained model restored from %s' %
                    (datetime.now(), pretrained_model_checkpoint_path))

# the name of the variable of interest is
# testtrain/testing/logits/abst_feats/Relu:0
print('Fetching all the images')

imgs = glob('/data/CNN/*')
print('Fetched %s images.' % fmt_num(len(imgs)))
total = len(imgs)

t = Thread(target=qworker, args=(enq_op, image_phds, imgs, sess))
t.start()

obtained = 0
print_inc = 1
start = time.time()

all_fns = []
all_logits = []
while obtained < total:
    imfns_n, logits_n = sess.run([imfns, logits])
    all_fns.append(imfns_n)
    all_logits.append(logits_n)
























