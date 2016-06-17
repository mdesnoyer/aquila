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


def batch_gen(pairs):
    pending_batches = []
    pkeys = list(pairs.keys())
    attempts = 0  # the number of attempts made on fetching a batch
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
                    if to_add == 0:
                        continue  # d'oh, this was the issue.
                    if len(pb) + to_add == BATCH_SIZE:
                        can_add = True
                    elif len(pb) + to_add < BATCH_SIZE - 1:
                        can_add = True
                    if can_add:
                        pb.add(i)
                        pb.add(j)
                        if len(pb) == BATCH_SIZE:
                            pending_batches.remove(pb)
                            attempts = 0
                            yield pb
                        break
                if not can_add:
                    pending_batches.append(set([i, j]))
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
    for m, i in enumerate(batch):
        for n, j in enumerate(batch):
            if (i, j) in labels:
                win_matrix[m, n, :] = labels[(i,j)][:DEMOGRAPHIC_GROUPS]
                win_matrix[n, m, :] = labels[(i,j)][DEMOGRAPHIC_GROUPS:]
    lb = [os.path.join(IMAGE_SOURCE, x) for x in batch]
    return lb, win_matrix.astype(np.uint8)


def bworker(pairs, labels, pyInQ):
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
    bg = batch_gen(pairs)
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
                pass
        with COUNT_LOCK:
            EPOCH_AND_BATCH_COUNT[1] += 1
            EPOCH_AND_BATCH_COUNT[2] += np.sum(win_matrix)

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
        if VERBOSE:
            print 'Enqueuing examples'
        try:
            sess.run(enq_op, feed_dict=feed_dict)
        except:
            if VERBOSE:
                print 'Enqueue fail error, returning'
            return


class InputManager(object):
    def __init__(self, image_phds, label_phds, conf_phds, tf_queue,
                 enqueue_op, num_epochs=20, num_qworkers=4):
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
        self.pairs, self.labels = read_in_data(DATA_SOURCE)
        self.tot_comparisons = reduce(lambda x, y: x + np.sum(y),
                                      self.labels.values(), 0)
        self.num_ex_per_epoch = self.tot_comparisons

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
                              args=(self.pairs, self.labels, self.in_q))
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