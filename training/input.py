"""
Creates an input manager to manage inputs to TensorFlow. Note the input
resolutions are hardcoded here.
"""

import numpy as np
import tensorflow as tf
import os
from threading import Thread
from threading import Event
from Queue import Queue
from Queue import Empty as QueueEmpty

# TODO: Eliminate testing shit below
# # ---------  FOR TESTING  ---------- #
# from scipy import sparse
# from scipy import io
# MPFN = '/repos/aquila/task_data/aggregated_data/idx_2_id'
# fnmap = dict()
# print 'Loading index to filename map'
# with open(MPFN, 'r') as f:
#     for line in f:
#         idx, fn = line.strip().split(',')
#         fnmap[int(idx)] = fn + '.jpg'
# print 'Creating win matrix - This takes a while'
# # wm_size = np.max(fnmap.keys()) + 1
# # win_matrix = sparse.lil_matrix((wm_size, wm_size), dtype=np.uint8)
# # data = np.load('/repos/aquila/task_data/aggregated_data/win_data.npy')
# # win_matrix[data[:,0], data[:,1]] = data[:,2]
# # win_matrix[data[:,1], data[:,0]] = data[:,3]
# win_matrix = io.mmread('/media/nick/d216eb37-b0e1-478c-b170-2270d7699ea21/repos'
#                        '/aquila/task_data/aggregated_data/test.mtx').astype(
#     np.uint8)
# print 'Done -- starting tensorflow stuffs'
# # --------- /FOR TESTING  ---------- #

VERBOSE = True  # whether or not the print all the shit you're doing


def get_enqueue_op(fn_phds, lab_phds, queue):
    """
    Obtains the TensorFlow batch enqueue operation.

    :param fn_phds: Filename TensorFlow placeholders (size=[batch_size])
    :param lab_phds: Label TensorFlow placeholders (size=[batch_size])
    :param queue: The TensorFlow input queue.
    :return: The enqueue operation.
    """
    im_tensors = []
    im_labels = []
    for fn_phd, lab_phd in zip(fn_phds, lab_phds):
        # read in the raw jpeg
        raw_im = tf.read_file(fn_phd)
        # convert to jpeg
        jpeg_im = tf.image.decode_jpeg(raw_im, channels=3)
        # random crop the image
        cropped_im = tf.random_crop(jpeg_im, [299, 299, 3])
        # random flip left/right
        im_tensor = tf.image.random_flip_left_right(cropped_im)
        im_tensors.append(tf.to_float(im_tensor))
        im_labels.append(tf.to_float(lab_phd))
    packed_ims = tf.pack(im_tensors)
    packed_labels = tf.pack(im_labels)
    enq_op = queue.enqueue_many([packed_ims, packed_labels])
    return enq_op


def _worker(win_matrix, filemap, imdir, batch_size, inq, outq, fn_phds,
            lab_phds, enq_op, sess):
    """
    The target of the worker threads. Manages the actual execution of the
    enqueuing of data.

    :param win_matrix: A sparse matrix or array X where X[i,j] = number
    of wins of item i over item j.
    :param filemap: A dictionary that maps indices to image filenames.
    :param imdir: The directory that contains the input images.
    :param batch_size: The size of a batch.
    :param inq: An input queue that stores the indicies of datapoints to
    measure. The input queue consists of tuples of indices (i, j).
    :param outq: The TensorFlow output queue.
    :param fn_phds: Filename TensorFlow placeholders.
    :param lab_phds: Label TensorFlow placeholders.
    :param enq_op: A tensorflow enqueue operation.
    :param sess: A TensorFlow session manager.
    :return: None
    """
    # get the enqueue operation
    # iterate until the queue is empty
    indices = np.zeros(batch_size).astype(int)
    while True:
        for sidx in np.arange(0, batch_size, 2):
            try:
                idx1, idx2 = inq.get(True, 5)
                indices[sidx] = idx1
                indices[sidx + 1] = idx2
            except QueueEmpty:
                if VERBOSE:
                    print 'Queue is empty, terminating'
                return
        print 'enqueuing', os.path.join(imdir, filemap[indices[0]])
        image_fns = [os.path.join(imdir, filemap[x]) for x in indices]
        image_labels = [win_matrix[x, indices].todense().A.squeeze() for x in indices]
        feed_dict = dict()
        # populate the feeder dictionary
        for fnp, fnd, labp, labd in zip(fn_phds, image_fns, lab_phds,
                                        image_labels):
            feed_dict[fnp] = fnd
            feed_dict[labp] = labd
        if VERBOSE:
            print 'Enqueuing', batch_size, 'examples'
        sess.run(enq_op, feed_dict=feed_dict)


class InputManager(object):
    def __init__(self, win_matrix, filemap,
                 imdir, tf_out, fn_phds,
                 lab_phds, enq_op,
                 batch_size=32, num_epochs=100,
                 num_threads=4):
        """
        Creates an object that manages the input to TensorFlow by managing a
        set of threads that enqueue batches of images. Handles all shuffling
        of data.

        NOTES:
            This spawns num_threads + 1 threads, with the last being the
            thread that's running the _Mgr classmethod, which manages enqueuing.

        :param win_matrix: A sparse matrix or array X where X[i,j] = number
        of wins of item i over item j.
        :param filemap: A dictionary that maps indices to image filenames.
        :param imdir: The directory that contains the input images.
        :param tf_out: The FIFO output queue.
        :param fn_phds: A list of TensorFlow placeholders of len batch_size (type: (tf.string, shape=[]))
        :param lab_phds: A list of TensorFlow placeholders of len batch_size (type: (tf.int32, shape=[batch_size]))
        :param enq_op: The TensorFlow enqueue operation.
        :param batch_size: The size of a batch.
        :param num_epochs: The number of epochs to run for.
        :param num_threads: The number of threads to spawn.
        :return: An instance of InputManager
        """
        self.win_matrix = win_matrix
        self.filemap = filemap
        self.imdir = imdir
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.outq = tf_out
        self.inq = Queue(maxsize=1024)
        self.num_threads = num_threads
        self.sess = sess
        self.fn_phds = fn_phds
        self.lab_phds = lab_phds
        self.enq_op = enq_op
        self.num_threads = num_threads
        a, b = self.win_matrix.nonzero()
        self.idxs = filter(lambda x: x[0] < x[1], zip(a, b))
        self.num_ex_per_epoch = len(idxs)
        self.n_examples = 0
        self.should_stop = Event()

    def start(self, sess):
        """
        Create & Starts all the threads
        """
        self.threads = [Thread(target=_worker,
                               args=(self.win_matrix, self.filemap, self.imdir, 
                                     self.batch_size, self.inq, self.outq, 
                                     self.fn_phds, self.lab_phds, self.enq_op, 
                                     sess))
                        for _ in range(self.num_threads)]
        self.mgr_thread = Thread(target=self._Mgr)
        for t in self.threads:
            t.daemon = True
            t.start()
        self.mgr_thread.start()

    def join(self):
        """
        Joins all threads
        """
        self.mgr_thread.join()

    def should_stop(self):
        """
        Indicates whether or not TensorFlow should halt
        """
        return self.should_stop.is_set()

    def _Mgr(self):
        """
        Manager class method. Should be started as a thread.
        """
        for epoch in range(self.num_epochs):
            np.random.shuffle(self.idxs)
            for idxs_pair in self.idxs:
                self.inq.put(idxs_pair)
                self.n_examples += 1
        print 'Enqueued all, total of %i' % self.n_examples
        for t in self.threads:
            t.join()
        self.should_stop.set()

# TODO: Eliminate testing shit below
# ---------  FOR TESTING  ---------- #
# sess = tf.InteractiveSession()
# outQ = tf.FIFOQueue(128, [tf.float32, tf.float32], shapes=[[299, 299, 3],
#                                                            [32]])
# imgr = InputManager(win_matrix, fnmap, imdir='/data/aquila_training_images',
#                     batch_size=32, num_epochs=5, tf_out=outQ, sess=sess,
#                     num_threads=1)
# imgr.start()
# while not imgr.should_stop():
# --------- /FOR TESTING  ---------- #


