"""
This script implements the training of Aquila, but does so purely with testing.
Furthermore, it does not attempt to run anything on the GPU, the num GPU is
purley to return the number of simulated GPUs. 

As a result, this script is substantially slower than the actual GPU
implementation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
from datetime import datetime
import os.path
import re
import time

import numpy as np
import tensorflow as tf

from net import aquila_model as aquila
from net.slim import slim
from config import *

from scipy import io
from scipy import sparse

IMG_DIR = '/data/images'
FILE_MAP_LOC = '/data/datasets/idx_2_id'
BATCH_SIZE = config.BATCH_SIZE
NUM_EPOCHS = config.NUM_EPOCHS
BATCH_SIZE *= num_gpus

save_step_size = 1  # the number of steps to take between fetching everything


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
    enq_op = queue.enqueue_many([packed_ims, packed_labels, tf.pack(fn_phds)])
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
                idx1, idx2 = inq.get(True, 30)
                indices[sidx] = idx1
                indices[sidx + 1] = idx2
            except QueueEmpty:
                if VERBOSE:
                    print 'Queue is empty, terminating'
                return
        image_fns = [os.path.join(imdir, filemap[x]) for x in indices]
        image_labels = [win_matrix[x, indices].todense().A.squeeze() for x in
                        indices]
        feed_dict = dict()
        # populate the feeder dictionary
        for fnp, fnd, labp, labd in zip(fn_phds, image_fns, lab_phds,
                                        image_labels):
            feed_dict[fnp] = fnd
            feed_dict[labp] = labd
        if VERBOSE:
            print 'Enqueuing', batch_size, 'examples'
        sess.run(enq_op, feed_dict=feed_dict)


def _single_win_map_worker(win_matrix, filemap, imdir, batch_size, inq, outq,
                           fn_phds, lab_phds, enq_op, sess):
    """
    The target of the worker threads. Manages the actual execution of the
    enqueuing of data. This worker is responsible for working under the
    single_win_mapping == True regime.

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
    indices = np.zeros(batch_size).astype(int)
    while True:
        image_labels = []
        for sidx in np.arange(0, batch_size, 2):
            try:
                idx1, idx2 = inq.get(True, 30)
                if not win_matrix[idx1, idx2]:
                    print 'GERROR'
                    return
                indices[sidx] = idx1
                indices[sidx + 1] = idx2
            except QueueEmpty:
                if VERBOSE:
                    print 'Queue is empty, terminating'
                return
            labs = np.zeros(batch_size).astype(int)
            labs[sidx + 1] = 1
            image_labels.append(labs)
            image_labels.append(np.zeros(batch_size).astype(int))
        image_fns = [os.path.join(imdir, filemap[x]) for x in indices]
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
                 batch_size, num_epochs=100,
                 num_threads=4,
                 debug_dir=None,
                 single_win_mapping=False):
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
        :param fn_phds: A list of TensorFlow placeholders of len batch_size
        (type: (tf.string, shape=[]))
        :param lab_phds: A list of TensorFlow placeholders of len batch_size
        (type: (tf.int32, shape=[batch_size]))
        :param enq_op: The TensorFlow enqueue operation.
        :param batch_size: The size of a batch.
        :param num_epochs: The number of epochs to run for.
        :param num_threads: The number of threads to spawn.
        :param debug_dir: If not None, it will store the ordering in which it
        stores the ordering of the inputs per epoch so errors may be re-created.
        :param single_win_mapping: If True, then it will repeatedly enqueue
        items about which we have more data.
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
        self.fn_phds = fn_phds
        self.lab_phds = lab_phds
        self.enq_op = enq_op
        self.num_threads = num_threads
        self.debug_dir = debug_dir
        self.single_win_mapping = single_win_mapping
        a, b = self.win_matrix.nonzero()
        # self.idxs = filter(lambda x: x[0] < x[1], zip(a, b))
        # why was i doing this? ^^^
        # self.idxs = zip(a[:16], b[:16])
        print 'Allocating indices'
        if not single_win_mapping:
            self.idxs = zip(a, b)
        else:
            self.idxs = []
            for a_, b_ in zip(a, b):
                if not win_matrix[a_, b_]:
                    print 'GERROR!'
                    return
                for _ in range(self.win_matrix[a_, b_]):
                    self.idxs.append([a_, b_])
        self.num_ex_per_epoch = len(self.idxs) * 2  # each entails 2 examples
        self.n_examples = 0
        self.should_stop = Event()
        self.mgr_thread = Thread(target=self._Mgr)
        self.mgr_thread.start()
        print 'Manager thread started'

    def start(self, sess):
        """
        Create & Starts all the threads
        """
        if self.single_win_mapping:
            targ = _single_win_map_worker
        else:
            targ = _worker
        self.threads = [Thread(target=targ,
                               args=(self.win_matrix, self.filemap, self.imdir,
                                     self.batch_size, self.inq, self.outq,
                                     self.fn_phds, self.lab_phds, self.enq_op,
                                     sess))
                        for _ in range(self.num_threads)]
        for t in self.threads:
            t.daemon = True
            t.start()

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
            if self.debug_dir is not None:
                fn = os.path.join(self.debug_dir, 'epoch_%i' % epoch)
                np.save(fn, self.idxs)
            for idxs_pair in self.idxs:
                self.inq.put(idxs_pair)
                self.n_examples += 1
        print 'Enqueued all, total of %i' % self.n_examples
        for t in self.threads:
            t.join()
        self.should_stop.set()


def _tower_loss(inputs, labels, scope):
    """
    Calculates the loss for a single tower, which is specified by scope.

    NOTES:
        Unlike in the original implementation for Inception, we will instead
        be dequeueing multiple batches for each tower.

    :param inputs: A BATCH_SIZE x 299 x 299 x 3 sized float32 tensor (images)
    :param labels: A [BATCH_SIZE x BATCH_SIZE] label matrix.
    :param scope: The tower name (i.e., tower_0)
    :returns: The total loss op.
    """

    # construct an instance of Aquila
    logits = aquila.inference(inputs, abs_feats, for_training=True,
                              restore_logits=restore_logits, scope=scope)
    # create the loss graph
    aquila.loss(logits, labels)

    # create the accuracy graph
    accuracy = aquila.accuracy(logits, labels)
    # accuracy_averages = tf.train.ExponentialMovingAverage(0.9,
    # name='accuracy')
    # accuracy_averages_op = accuracy_averages.apply([accuracy])

    # fetch the actual losses, both the ranknet and the regularization loss
    # functions.
    losses = tf.get_collection(slim.losses.LOSSES_COLLECTION, scope)
    regularization_losses = tf.get_collection(
                                tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = tf.add_n(losses + regularization_losses, name='total_loss')
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    loss_averages_op = loss_averages.apply(losses + [total_loss, accuracy])
    for l in losses + [total_loss]:
        loss_name = re.sub('%s_[0-9]*/' % aquila.TOWER_NAME, '', l.op.name)
        tf.scalar_summary(loss_name +' (raw)', l)
        tf.scalar_summary(loss_name, loss_averages.average(l))
    tf.scalar_summary('accuracy (raw)', accuracy)
    tf.scalar_summary('accuracy', loss_averages.average(accuracy))
    with tf.control_dependencies([loss_averages_op]):
        total_loss = tf.identity(total_loss)
    return total_loss, logits


def _average_gradients(tower_grads):
    """
    Calculate the average gradient for each shared variable across all towers.

    NOTES:
        This function provides a synchronization point across all towers.

    :param tower_grads: List of lists of (gradient, variable) tuples. The outer
    list is over individual gradients. The inner list is over the gradient
    calculation for each tower.
    :returns: List of pairs of (gradient, variable) where the gradient has been
    averaged across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(0, grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def train(inp_mgr, ex_per_epoch):
    """
    Trains the network for some number of epochs.

    :param inp_mgr: An instance of the input manager.
    :param num_epochs: The number of epochs to run for.
    :param ex_per_epoch: The number of examples per epoch.
    """
    global_step = tf.get_variable(
                'global_step', [],
                initializer=tf.constant_initializer(0), trainable=False)

    num_batches_per_epoch = ex_per_epoch / BATCH_SIZE
    max_steps = int(num_batches_per_epoch * NUM_EPOCHS)
    decay_steps = int(num_batches_per_epoch * num_epochs_per_decay)
    lr = tf.train.exponential_decay(initial_learning_rate,
                                    global_step,
                                    decay_steps,
                                    learning_rate_decay_factor,
                                    staircase=True)
    # Create an optimizer that performs gradient descent.
    opt = tf.train.RMSPropOptimizer(lr, RMSPROP_DECAY,
                                    momentum=RMSPROP_MOMENTUM,
                                    epsilon=RMSPROP_EPSILON)

    # Get images and labels for ImageNet and split the batch across GPUs.
    assert BATCH_SIZE % num_gpus == 0, (
            'Batch size must be divisible by number of GPUs')
    split_batch_size = int(BATCH_SIZE / num_gpus)

    input_summaries = copy.copy(tf.get_collection(tf.GraphKeys.SUMMARIES))

    # Calculate the gradients for each model tower.
    tower_grads = []
    tower_inputs = []
    tower_input_names = []
    tower_labels = []
    tower_label_names = []
    tower_filenames = []
    tower_filename_names = []
    tower_logits = []
    tower_logit_names = []
    op_names = ['null', 'loss']
    for i in xrange(num_gpus):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('%s_%d' % (aquila.TOWER_NAME, i)) as scope:
                # Calculate the loss for one tower of the ImageNet model. This
                # function constructs the entire ImageNet model but shares the
                # variables across all towers.
                tow_pfx = '%s_%d' % (aquila.TOWER_NAME, i)
                inputs, labels, filenames = inp_mgr.outq.dequeue_many(split_batch_size)
                tf.scalar_summary('input_queue_size', inp_mgr.outq.size())
                m_4d_ = tf.reshape(labels, [1, split_batch_size,
                                            split_batch_size, 1])
                loss, _logits = _tower_loss(inputs, labels, scope)

                # Reuse variables for the next tower.
                tf.get_variable_scope().reuse_variables()

                # Retain the summaries from the final tower.
                summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                # Retain the Batch Normalization updates operations only from the
                # final tower. Ideally, we should grab the updates from all towers
                # but these stats accumulate extremely fast so we can ignore the
                # other stats from the other towers without significant detriment.
                batchnorm_updates = tf.get_collection(
                        slim.ops.UPDATE_OPS_COLLECTION, scope)

                # Calculate the gradients for the batch of data on this ImageNet
                # tower.
                grads = opt.compute_gradients(loss)

                # Keep track of the gradients across all towers.
                tower_grads.append(grads)

                tower_inputs.append(inputs)
			    tower_input_names.append(tow_pfx + '_inputs')
			    tower_labels.append(labels)
			    tower_label_names.append(tow_pfx + '_labels')
			    tower_filenames.append(filenames)
			    tower_filename_names.append(tow_pfx + '_filenames')
			    tower_logits.append(_logits)
			    tower_logit_names.append(tow_pfx + '_logits')
                tower_input_names.append(tow_pfx + '_inputs')



    # We must calculate the mean of each gradient. Note that this is the
    # synchronization point across all towers.
    grads = _average_gradients(tower_grads)

    # Apply the gradients to adjust the shared variables.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Track the moving averages of all trainable variables.
    # Note that we maintain a "double-average" of the BatchNormalization
    # global statistics. This is more complicated then need be but we employ
    # this for backward-compatibility with our previous models.
    variable_averages = tf.train.ExponentialMovingAverage(
            aquila.MOVING_AVERAGE_DECAY, global_step)

    # Another possibility is to use tf.slim.get_variables().
    variables_to_average = (tf.trainable_variables() +
                            tf.moving_average_variables())
    variables_averages_op = variable_averages.apply(variables_to_average)

    # Group all updates to into a single train op.
    batchnorm_updates_op = tf.group(*batchnorm_updates)
    train_op = tf.group(apply_gradient_op, variables_averages_op,
                                            batchnorm_updates_op)

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    # Start running operations on the Graph. allow_soft_placement must be set to
    # True to build towers on GPU, as some of the ops do not have GPU
    # implementations.
    sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=log_device_placement))
    sess.run(init)

    # start the input manager?
    inp_mgr.start(sess)

    tower_grads = []
    tower_inputs = []
    tower_labels = []
    tower_filenames = []
    tower_logits = []

    # summary_writer = tf.train.SummaryWriter(
    #             train_dir, graph_def=sess.graph.as_graph_def(add_shapes=True))
    print('%s: Model running for %i iterations' %
          (datetime.now(), max_steps))
    for step in xrange(max_steps):
    	start_time = time.time()
    	if step % save_step_size:
        	_, loss_value = sess.run([train_op, loss])
        else:
        	storeops = [train_op, loss] + tower_grads + tower_inputs
        	storeops += tower_labels + tower_filesnames + tower_logits
        	vals = sess.run(storeops)

        duration = time.time() - start_time

        if np.isnan(loss_value):
            raise Exception('Model diverged with loss = NaN on epoch %i' % step)

        if step % 1 == 0:
            examples_per_sec = BATCH_SIZE / float(duration)
            format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; '
                          '%.3f sec/batch)')
            print(format_str % (datetime.now(), step, loss_value,
                                                    examples_per_sec, duration))

if config.subset == 'train':
    WIN_MATRIX_LOC = '/data/datasets/train/win_matrix.mtx'
else:
    WIN_MATRIX_LOC = '/data/datasets/test/win_matrix.mtx'

fnmap = dict()
print 'Loading index to filename map'
with open(FILE_MAP_LOC, 'r') as f:
    for line in f:
        idx, fn = line.strip().split(',')
        fnmap[int(idx)] = fn + '.jpg'
print 'Loading win matrix'
win_matrix = sparse.lil_matrix(io.mmread(WIN_MATRIX_LOC).astype(np.uint8))

outQ = tf.FIFOQueue(BATCH_SIZE*16, [tf.float32, tf.float32, tf.string], 
					shapes=[[299, 299, 3], [BATCH_SIZE], [BATCH_SIZE]])
fn_phds = [tf.placeholder(tf.string, shape=[]) for _ in range(BATCH_SIZE)]
lab_phds = [tf.placeholder(tf.int32, 
                           shape=[BATCH_SIZE]) for _ in range(BATCH_SIZE)]
enq_op = get_enqueue_op(fn_phds, lab_phds, outQ)

imgr = InputManager(win_matrix, fnmap, IMG_DIR, outQ, fn_phds, lab_phds,
                    enq_op, BATCH_SIZE, num_epochs=NUM_EPOCHS, num_threads=1,
                    debug_dir='/data/training_epoch_sequence',
                    single_win_mapping=True)

aquila_train.train(imgr, imgr.num_ex_per_epoch)