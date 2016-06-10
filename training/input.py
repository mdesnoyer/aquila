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
from Queue import Empty as QueueEmpty
import locale
from collections import defaultdict as ddict
from config import *
from PIL import Image

try:
    # for linux
    locale.setlocale(locale.LC_ALL, 'en_US.utf8')
except:
    # for mac
    locale.setlocale(locale.LC_ALL, 'en_US')


VERBOSE = False  # whether or not the print all the shit you're doing
EPOCH_AND_BATCH_COUNT = [0, 0]  # [n_epochs, n_batches]
SHOULD_STOP = Event()
COUNT_LOCK = Lock()

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
    print 'Total read: %s' % locale.format("%d", n, grouping=True)
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
    for image_phd, lab_phd, conf_phd in zip(image_phds, label_phds, conf_phds):
        # read in the raw jpeg
        raw_im = tf.read_file(image_phd)
        # convert to jpeg
        jpeg_im = tf.image.decode_jpeg(raw_im, channels=3)
        # random crop the image
        cropped_im = tf.random_crop(jpeg_im, [299, 299, 3])
        # random flip left/right
        im_tensor = tf.image.random_flip_left_right(cropped_im)
        im_tensors.append(im_tensor)
        im_labels.append(lab_phd)
        im_conf.append(conf_phd)
    packed_ims = tf.pack(im_tensors)
    packed_labels = tf.pack(im_labels)
    packed_conf = tf.pack(im_conf)
    enq_op = queue.enqueue_many([packed_ims, packed_labels, packed_conf])
    return enq_op

################################################################################
################################################################################
#                       WORKER FUNCTIONS                                       #
################################################################################
################################################################################

################################################################################
# BATCH WORKER & HELPERS
################################################################################


def _yield_sets(i, pairs, pending_batches=[]):
    """
    Given pending batches, and a pair group (i -> pairs), this will construct
    and sequentially yield appropriate batches. At most two incomplete batches
    are generated.

    NOTES:
        WARNING:
            Modifies pending batches in-place.

    Args:
        i: The first pair item.
        pairs: A list of items paired with i
        pending_batches: Incomplete batches.

    Returns: An iterator over the assembled batches.
    """

    def get_next_batch():
        if len(pending_batches):
            return pending_batches.pop()
        return set()

    batch = get_next_batch()
    while pairs:
        if len(batch) == BATCH_SIZE:
            yield batch
            batch = get_next_batch()
            continue
        if len(batch | set(pairs) | set((i,))) == BATCH_SIZE - 1:
            # then it will produce aliasing.
            batch.add(i)
            while len(batch) < BATCH_SIZE - 2:
                batch.add(pairs.pop())
            yield batch
            batch = get_next_batch()
            continue
        batch.add(i)
        batch.add(pairs.pop())
        if len(batch) == BATCH_SIZE:
            yield batch
            batch = get_next_batch()
    if batch:
        yield batch
    while pending_batches:
        yield pending_batches.pop()


def batch_gen(pairs):
    cur_batch = set()
    pending_batches = []
    while True:
        np.random.shuffle(pairs)
        for i in pairs:
            pair_items = pairs[i]
            constructed_batches = _yield_sets(i, pair_items, pending_batches)
            pending_batches = []
            for batch in constructed_batches:
                if len(batch) != BATCH_SIZE:
                    pending_batches.append(batch)
                else:
                    yield batch
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
    win_matrix = np.zeros((BATCH_SIZE, BATCH_SIZE, DEMOGRAPHIC_GROUPS))
    lb = list(batch)
    for m, i in enumerate(lb):
        for n, j in enumerate(lb):
            if (i, j) in labels:
                win_matrix[m, n, :] = labels[(i,j)][:DEMOGRAPHIC_GROUPS]
                win_matrix[n, m, :] = labels[(i,j)][DEMOGRAPHIC_GROUPS:]
    return lb, win_matrix


def bworker(data_location, pyInQ):
    """
    Batch Worker. Designed to work asyncronously, continuously inserts new
    batches into the queue as tuples of (images, win_matrix), where win_matrix
    is just a different representation of a subset of the labels.

    Args:
        data_location: The source of the data file.
        pyInQ: An input queue to store tuples of (image_fns, win_matrix,
        confidence_matrix)

    Returns: None.
    """
    pairs, labels = read_in_data(data_location)
    bg = batch_gen(pairs)
    while True:
        if SHOULD_STOP.is_set():
            return
        next_batch = bg.next()
        batch, win_matrix = gen_labels(next_batch, labels)
        # TODO: generate the confidence matrix
        confidence_matrix = np.ones_like(win_matrix).astype(np.int8)
        pyInQ.put((batch, win_matrix, confidence_matrix))
        with COUNT_LOCK:
            EPOCH_AND_BATCH_COUNT[1] += 1

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

    Args:
        pyInQ: Input queue, consisting of tuples of (batch_images, win_matrix)
        pyOutQ:

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
        if VERBOSE:
            print 'Enqueuing examples'
        sess.run(enq_op, feed_dict=feed_dict)


class InputManager(object):
    def __init__(self, tfOutQ, data_location, image_phds, label_phds, conf_phds,
                 tf_queue, num_epochs=20, num_qworkers=4):
        self.out_Q = tfOutQ
        self.image_phds = image_phds
        self.label_phds = label_phds
        self.conf_phds = conf_phds
        self.epochs = num_epochs
        self.stopper = SHOULD_STOP
        self.bworker = None
        self.qworkers = []
        self.num_qworkers = num_qworkers
        self.data_location = data_location
        self.in_q = Queue()
        self.enq_op = get_enqueue_op(image_phds, label_phds, conf_phds,
                                     tf_queue)

    def start(self, sess):
        self.bworker = Thread(target=bworker,
                              args=(self.data_location, self.in_q))
        self.bworker.start()
        for t in range(self.num_qworkers):
            qw = Thread(target=pyqworker(self.in_q, sess, self.enq_op,
                                         self.image_phds, self.label_phds,
                                         self.conf_phds))
            qw.start()
            self.qworkers.append(qw)

    def should_stop(self):
        with COUNT_LOCK:
            if EPOCH_AND_BATCH_COUNT[0] >= self.epochs:
                self.stopper.set()
                return True
        return False

    def stop(self):
        self.bworker.join()
        for qw in self.qworkers:
            qw.join()