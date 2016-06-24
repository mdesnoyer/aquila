"""
Creates an input manager to manage inputs to TensorFlow. Note the input
resolutions are hardcoded here.
"""

import numpy as np
import tensorflow as tf
import os
from threading import Thread
from threading import Event
from threading import Lock
from Queue import Queue
import locale
from collections import defaultdict as ddict
from config import *
from PIL import Image
from time import sleep
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python import ops

try:
    # for linux
    locale.setlocale(locale.LC_ALL, 'en_US.utf8')
except:
    # for mac
    locale.setlocale(locale.LC_ALL, 'en_US')


VERBOSE = False  # whether or not the print all the shit you're doing
EPOCH_AND_BATCH_COUNT = [0, 0, 0]  # [n_epochs, n_batches, n_comparisons]
SHOULD_STOP = Event()
COUNT_LOCK = Lock()
# the bin freqs are added to the bins whenever an unkonwn individual appears
BIN_FREQ = np.array([0.16, 0.10, 0.15, 0.05, 0.03, 0.14, 0.06, 0.15, 0.08,
                     0.07])


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

def get_confidence(im1, im2):
    """
    Returns the confidence in the pairing for im1 and im2.

    Args:
        im1: The first image.
        im2: The second image.

    Returns: The confidence, as a float between 0 and 1.
    """
    return 1.0  # for now, we're going to be completely confident


def read_in_data(data_location):
    """
    Creates the objects for feeding data.

    Args:
        data_location: A string pointing to a file.

    Returns:
        pairs: A dictionary, indexed by `image`, that points to a set of all
        images that `image` has been compared to. Note that an image will
        only have a key in `pairs` if it is the lexicographically first image
        in at least one pair.

        labels: A dictionary, indexed by a sorted tuple of images (`image1`,
        `image2`). This points to a numpy array of integers, where indices
        0-8 indicate the number of wins by `image1` over `image2`, binned by
        demographics, while indices 9-15 indicate the number of times
        `image2` has beaten `image1`, binned by demographics.
    """
    pairs = ddict(lambda: set())
    labels = dict()
    with open(data_location, 'r') as f:
        for n, line in enumerate(f):
            cur_data = line.strip().split(',')
            img_a = cur_data[0]
            img_b = cur_data[1]
            outcomes = [int(x) for x in cur_data[2:]]
            pairs[img_a].add(img_b)
            labels[(img_a, img_b)] = np.array(outcomes).astype(int)
            if not n % 1000:
                print 'Total read: %s' % locale.format("%d", n, grouping=True)
            if SUBSET_SIZE is not None:
                if SUBSET_SIZE * 5 < n:
                    break
    print 'Total read: %s' % locale.format("%d", n, grouping=True)
    if SUBSET_SIZE is not None:
        print 'Selecting subset of size %i' % SUBSET_SIZE
        lkeys = labels.keys()
        cidxs = np.random.choice(len(lkeys), SUBSET_SIZE / 2,
                                 replace=False)
        chosen = []
        for cidx in cidxs:
            chosen.append(lkeys[cidx])
        cpairs = ddict(lambda: set())
        clabels = dict()
        for a, b in chosen:
            clabels[(a, b)] = labels[(img_a, img_b)]
            cpairs[a].add(b)
        return cpairs, clabels
    return pairs, labels


def get_enqueue_op(image_phds, label_phds, conf_phds, queue):
    """
    Obtains the TensorFlow batch enqueue operation.

    :param fn_phds: Filename TensorFlow placeholders (size=[batch_size])
    :param lab_phds: Label TensorFlow placeholders (size=[batch_size])
    :param queue: The TensorFlow input queue.
    :return: The enqueue operation.
    """
    im_tensors = []
    im_labels = []
    im_conf = []
    # process the mean channel values to integers so that we can still pass
    # messages as integers
    mcv = np.array(MEAN_CHANNEL_VALS).round().astype(np.float32)
    channel_mean_tensor = tf.constant(mcv)

    for image_phd, lab_phd, conf_phd in zip(image_phds, label_phds, conf_phds):
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
        im_labels.append(lab_phd)
        im_conf.append(conf_phd)
    packed_ims = tf.to_float(tf.pack(im_tensors))
    packed_labels = tf.to_float(tf.pack(im_labels))
    packed_conf = tf.to_float(tf.pack(im_conf))
    packed_fns = tf.pack(image_phds)
    enq_op = queue.enqueue_many([packed_ims, packed_labels,
                                 packed_conf, packed_fns])
    return enq_op

################################################################################
################################################################################
#                       WORKER FUNCTIONS                                       #
################################################################################
################################################################################

################################################################################
# BATCH WORKER & HELPERS
################################################################################


def _batch_gen(pairs, training=True):
    pending_batches = []
    pkeys = list(pairs.keys())
    attempts = 0  # the number of attempts made on fetching a batch
    max_pb = 100
    while True:
        np.random.shuffle(pkeys)
        for i in pkeys:
            pair_items = list(pairs[i])
            np.random.shuffle(pair_items)
            for j in pair_items:
                can_add = False
                for pb in pending_batches:
                    if (attempts + 1) % 100 == 0:
                        print 'Warning! Made %i attempts to generate a batch,' \
                              ' with %i pending ' \
                              'batches' % (attempts, len(pending_batches))
                    attempts += 1
                    to_add = (i not in pb) + (j not in pb)
                    if len(pb) + to_add > BATCH_SIZE:
                        continue
                    pb.add(i)
                    pb.add(j)
                    if len(pb) == BATCH_SIZE:
                        pending_batches.remove(pb)
                        attempts = 0
                        can_add = True
                        yield pb
                    break
                if not can_add:
                    pending_batches.append({i, j})
                if len(pending_batches) > max_pb:
                    pb = pending_batches[0]
                    if len(pb) == (BATCH_SIZE - 1):
                        _ = pending_batches.pop(0)
                        pb.add(None)
                        attempts = 0
                        yield pb
        if training:
            with COUNT_LOCK:
                EPOCH_AND_BATCH_COUNT[0] += 1


def _get_int_sz(targ, a_b_s):
    """ returns 0 if {a, b} is in the targ, 1 if one already is, 2 if neither
    are, and float("inf") if they can't fit"""
    tlen = len(targ)
    toadd = len(targ & a_b_s)
    if (toadd + tlen) > BATCH_SIZE:
        return float("inf")
    return toadd


def _add_pair(pending, a, b):
    """ adds {a, b} to a set in pending. if a completed set is
    generated, it returns <completed set> and removes it from pending.
    if {a, b} cant be added to any set, it creates a new one for them and
    adds it to pending """
    a_b_s = {a, b}
    if not len(pending):
        pending.append(a_b_s)
        return
    dists = [_get_int_sz(x, a_b_s) for x in pending]
    idx = np.argmin(dists)
    idxv = dists[idx]
    ival = float("inf")
    if idxv == ival:
        pending.append(a_b_s)
    else:
        pending[idx].update(a_b_s)
        if len(pending[idx]) == BATCH_SIZE:
            return pending.pop(idx)
    return None


def _get_closest_pending(pending):
    """ pops a nearly complete pending batch from the pending batch list and
    adds the null image and returns it. returns none if no nearly complete
    pending batches exist """
    for idx in range(len(pending)):
        if len(pending[idx]) == (BATCH_SIZE - 1):
            pend = pending.pop(idx)
            pend.add(None)
            return pend
    return None


def batch_gen(pairs, training=True):
    pending_batches = []
    pkeys = list(pairs.keys())
    max_pb = 100
    uni_ims = set()
    for a in pairs:
        for b in pairs[a]:
            uni_ims.add(a)
            uni_ims.add(b)
    num_uni_ims = len(uni_ims)
    seen_inc = 100
    cseen = 0
    stat_str = "\t%s images seen (%.2f%%) over %s comparisons, epoch %s, " \
               "batch %s"
    while True:
        np.random.shuffle(pkeys)
        for i in pkeys:
            pair_items = list(pairs[i])
            np.random.shuffle(pair_items)
            for j in pair_items:
                pos = _add_pair(pending_batches, i, j)
                if pos:
                    uni_ims.difference_update(pos)
                    cseen += 1
                    if (cseen % seen_inc) == 0 and training:
                        # [n_epochs, n_batches, n_comparisons]
                        n_seen = num_uni_ims - len(uni_ims)
                        seen_rat = 100. * float(n_seen) / num_uni_ims
                        nseenstr = locale.format("%d", n_seen, grouping=True)
                        axx, bxx, cxx = \
                            EPOCH_AND_BATCH_COUNT
                        axx = locale.format("%d", axx, grouping=True)
                        bxx = locale.format("%d", bxx, grouping=True)
                        cxx = locale.format("%d", cxx, grouping=True)
                        c_stat_str = stat_str % (nseenstr, seen_rat, cxx,
                                                 axx, bxx)
                        print c_stat_str
                        cseen = 0
                    yield pos
                elif len(pending_batches) > max_pb:
                    pos = _get_closest_pending(pending_batches)
                    if pos:
                        uni_ims.difference_update(pos)
                        cseen += 1
                        if (cseen % seen_inc) == 0 and training:
                            n_seen = num_uni_ims - len(uni_ims)
                            seen_rat = 100. * float(n_seen) / num_uni_ims
                            nseenstr = locale.format("%d", n_seen, grouping=True)
                            axx, bxx, cxx = \
                                EPOCH_AND_BATCH_COUNT
                            axx = locale.format("%d", axx, grouping=True)
                            bxx = locale.format("%d", bxx, grouping=True)
                            cxx = locale.format("%d", cxx, grouping=True)
                            c_stat_str = stat_str % (nseenstr, seen_rat, cxx,
                                                     axx, bxx)
                            print c_stat_str
                            cseen = 0
                        yield pos
        if training:
            with COUNT_LOCK:
                EPOCH_AND_BATCH_COUNT[0] += 1


def gen_labels(batch, labels):
    '''
    Generates a label, given a batch and the labels dictionary.

    Args:
        batch: The current batch, as a set.
        labels: The labels dictionary.

    Returns: The batch, and a win matrix, of size batch x batch x demographic
    groups
    '''
    dgs = DEMOGRAPHIC_GROUPS
    win_matrix = np.zeros((BATCH_SIZE, BATCH_SIZE, dgs - 1))
    for m, i in enumerate(batch):
        for n, j in enumerate(batch):
            if (i, j) in labels:
                mn = labels[(i,j)][:dgs-1] + labels[(i,j)][dgs] * BIN_FREQ
                win_matrix[m, n, :] = mn
                nm = labels[(i,j)][dgs:-1] + labels[(i,j)][-1] * BIN_FREQ
                win_matrix[n, m, :] = nm
    lb = []
    for x in batch:
        if x is not None:
            lb.append(os.path.join(IMAGE_SOURCE, x))
        else:
            lb.append(NULL_IMAGE)
    return lb, win_matrix


def bworker(pairs, labels, pyInQ, training=True):
    """
    Batch Worker. Designed to work asynchronously, continuously inserts new
    batches into the queue as tuples of (images, win_matrix), where win_matrix
    is just a different representation of a subset of the labels.

    Args:
        data_location: The source of the data file.
        pyInQ: An input queue to store tuples of (image_fns, win_matrix,
        confidence_matrix)

    Returns: None.
    """
    bg = batch_gen(pairs, training=training)
    while True:
        next_batch = bg.next()
        batch, win_matrix = gen_labels(next_batch, labels)
        # TODO: generate the confidence matrix
        confidence_matrix = np.ones_like(win_matrix).astype(float)
        while True:
            if SHOULD_STOP.is_set():
                return
            try:
                pyInQ.put((batch, win_matrix, confidence_matrix),
                          block=True, timeout=10)
                break
            except:
                if VERBOSE:
                    print 'Failed to place batch in batch queue'
        if VERBOSE:
                print 'Placed batch in batch queue'
        if training:
            with COUNT_LOCK:
                EPOCH_AND_BATCH_COUNT[1] += 1
                EPOCH_AND_BATCH_COUNT[2] += np.sum(np.sum(win_matrix, 3) > 0)

################################################################################
# PYTHON QUEUE WORKER & HELPERS
################################################################################


def _prep_image(fn):
    """
    Retrieves and preprocesses an image.

    Args:
        fn: An image filename.

    Returns: The image as a 299x299 RGB image.
    """
    try:
        i = Image.open(fn)
    except:
        return None
    w, h = i.size
    cx = np.random.randint(0, (w - 299)+1)
    cy = np.random.randint(0, (h - 299)+1)
    c = i.crop((cx, cy, cx+299, cy+299))  # produce a random crop
    # do a random flip
    if np.random.rand() > 0.5:
        c.transpose(Image.FLIP_LEFT_RIGHT)
    return np.array(c)


def pyqworker(pyInQ, sess, enq_op, image_phds, label_phds, conf_phds):
    """
    Dequeues batch requests from python's input queue, reads the images,
    and then bundles them (along with the win matrix) to the output queue.

    Returns:

    """
    feed_dict = dict()
    while True:
        while True:
            if SHOULD_STOP.is_set():
                return
            try:
                item = pyInQ.get(block=True, timeout=10)
                break
            except:
                pass
        batch_images, win_matrix, conf_matrix = item
        for i in range(BATCH_SIZE):
            feed_dict[image_phds[i]] = batch_images[i]
            feed_dict[label_phds[i]] = win_matrix[i, :, :].squeeze()
            feed_dict[conf_phds[i]] = conf_matrix[i, :, :].squeeze()
        attempts = 0
        while True:
            try:
                attempts += 1
                sess.run(enq_op, feed_dict=feed_dict)
                attempts = 0
                if VERBOSE:
                    print 'Enqueued examples'
                break
            except Exception, e:
                if VERBOSE:
                    print 'Enqueue fail error:', e.message
                    sleep(10)
                if SHOULD_STOP.is_set():
                    if VERBOSE:
                        print 'Should stop is set, returning'
                        return
            if attempts >= 5:
                print 'Attempt limit has been exceeded. Batch:'
                for i in batch_images:
                    print '\t', i
                break


class InputManager(object):
    def __init__(self, image_phds, label_phds, conf_phds, tf_queue,
                 enqueue_op, num_epochs=20, num_qworkers=4,
                 data_source=DATA_SOURCE, is_training=True):
        """
        The input manager class

        :param image_phds: Image filename placeholders, a list of BATCH_SIZE
        tensorflow string placeholders
        :param label_phds: Label placeholders (uint8)
        :param conf_phds: Confidence placeholders (floats)
        :param tf_queue: The tensorflow input queue (output queue from Input
        manager's perspective). Should store:
            [tf.float, 299 x 299 x 3]                    < the images
            [tf.uint8, BATCH_SIZE x DEMOGRAPHIC_GROUPS]  < the labels
            [tf.float, BATCH_SIZE x DEMOGRAPHIC_GROUPS]  < the confidences
        :param num_epochs: The number of epochs
        :param num_qworkers: The number of queue workers.
        :param data_source: Where the data is stored.
        :param is_training: Whether or not this input manager manages a
        training queue.
        :return:
        """
        self.image_phds = image_phds
        self.label_phds = label_phds
        self.conf_phds = conf_phds
        self._epochs = num_epochs
        self.stopper = SHOULD_STOP
        self.bworker = None
        self.qworkers = []
        self.EPOCH_AND_BATCH_COUNT = EPOCH_AND_BATCH_COUNT
        self.num_qworkers = num_qworkers
        self.in_q = Queue(maxsize=BATCH_SIZE * num_gpus * 2)
        self.tf_queue = tf_queue
        self.enq_op = enqueue_op
        self.pairs, self.labels = read_in_data(data_source)
        self.tot_comparisons = reduce(lambda x, y: x + np.sum(y),
                                      self.labels.values(), 0)
        self.num_ex_per_epoch = self.tot_comparisons * 1.5
        self.is_training = is_training

    @property
    def epochs(self):
        with COUNT_LOCK:
            return self.EPOCH_AND_BATCH_COUNT[0]

    @property
    def batches(self):
        with COUNT_LOCK:
            return self.EPOCH_AND_BATCH_COUNT[1]

    @property
    def comparisons(self):
        with COUNT_LOCK:
            return self.EPOCH_AND_BATCH_COUNT[2]

    def start(self, sess):
        self.sess = sess
        self.bworker = Thread(target=bworker,
                              args=(self.pairs, self.labels, self.in_q,
                                    self.is_training))
        self.bworker.daemon = True
        self.bworker.start()
        for t in range(self.num_qworkers):
            qw = Thread(target=pyqworker,
                        args=(self.in_q, sess, self.enq_op, self.image_phds,
                              self.label_phds, self.conf_phds))
            qw.daemon = True
            qw.start()
            self.qworkers.append(qw)
        if VERBOSE:
            print 'Image Manager Started'

    def should_stop(self):
        if self.is_training:
            with COUNT_LOCK:
                if self.EPOCH_AND_BATCH_COUNT[0] >= self._epochs:
                    self.stopper.set()
                    return True
        return False

    def stop(self):
        if VERBOSE:
            print 'Stopping'
        self.stopper.set()
        print 'Attempting to stop tensorflow queue'
        close_op = self.tf_queue.close(cancel_pending_enqueues=True)
        self.sess.run(close_op)
        for qw in self.qworkers:
            qw.join()
            if VERBOSE:
                print 'Stopped a queue worker'
        self.bworker.join()
        if VERBOSE:
            print 'Stopped the batch worker'
