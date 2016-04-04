"""
This will generate and train the network. Mainly, this is to test the input
and loss functions.
"""
import numpy as np
import tensorflow as tf 
import threading
from glob import glob

N_EPOCHS = 20000
BATCH_SIZE = 16
IM_H = 128
IM_W = 128
LEARNING_RATE = 0.01
DISPLAY_STEP = 20
N_INPUT = IM_H * IM_W
STD_DEV = 0.01
WLFN = 'win_data/winlist.npy'
MPFN = 'win_data/id_2_fn'


# tf_fn_phd = tf.placeholder(tf.string, shape=[])
# tf_lab_phd = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
# tf_fn_q = tf.FIFOQueue(128, [tf.float32], shapes=[[IM_H, IM_W, 1]])
# tf_lab_q = tf.FIFOQueue(128, [tf.float32], shapes=[[BATCH_SIZE]])
tf_q_grp = tf.FIFOQueue(128, [tf.float32, tf.float32, tf.int8], shapes=[[IM_H, IM_W, 1], [BATCH_SIZE], [1]])
# tf_lab_enqueue_op = tf_lab_q.enqueue([tf.to_float(tf_lab_phd)])


def read_images_in(tf_fn_phd):
    """
    Takes a raw filename and enqueues the file specified

    fn: a placeholder
    """
    raw_im_ = tf.read_file(tf_fn_phd)
    im_ = tf.image.decode_jpeg(raw_im_)
    r_im_ = tf.image.resize_images(im_, IM_H, IM_W)
    return r_im_

# read all the files in for validation
ims = glob('images/*.jpg')
ims = np.sort(ims)
tf_ims = []
for im in ims:
    tf_ims.append(read_images_in(tf.constant(im)))
val_X = tf.pack(tf_ims)

data = np.load(WLFN)
fnmap = dict()
with open(MPFN, 'r') as f:
    for line in f:
        idx, fn = line.strip().split(',')
        fnmap[int(idx)] = fn


def get_fn_enqueue_op(tf_fn_phds, labels, idxs, queue):
    """
    Takes a raw filename and enqueues the file specified

    fn: a placeholder
    """
    im_tensors = []
    im_labels = []
    im_idxs = []
    for tf_fn_phd in tf_fn_phds:
        raw_im_ = tf.read_file(tf_fn_phd)
        im_ = tf.image.decode_jpeg(raw_im_)
        cur_tensor = tf.image.resize_images(im_, IM_H, IM_W)
        im_tensors.append(tf.to_float(cur_tensor))
    for lab_phd in labels:
        im_labels.append(tf.to_float(lab_phd))
    for idx in idxs:
        im_idxs.append(idx)
    stuffs1 = tf.pack(im_tensors)
    stuffs2 = tf.pack(im_labels)
    stuffs3 = tf.pack(im_idxs)
    #enq_op = queue.enqueue_many((tf.pack(im_tensors), tf.pack(labels)))
    enq_op = queue.enqueue_many([stuffs1, stuffs2, stuffs3])
    return enq_op

def get_packed_imlabs(tf_fn_phds, labels):
    im_tensors = []
    for tf_fn_phd in tf_fn_phds:
        raw_im_ = tf.read_file(tf_fn_phd)
        im_ = tf.image.decode_jpeg(raw_im_)
        cur_tensor = tf.image.resize_images(im_, IM_H, IM_W)
        im_tensors.append(tf.to_float(cur_tensor))
    return tf.pack(im_tensors), tf.pack(labels)    

placeholders = [tf.placeholder(tf.string, shape=[]) for _ in range(BATCH_SIZE)]
labels = [tf.placeholder(tf.int32, shape=[BATCH_SIZE]) for _ in range(BATCH_SIZE)]
idx_placeholders = [tf.placeholder(tf.int8, shape=[1]) for _ in range(BATCH_SIZE)]
#stuffs1, stuffs2 = get_packed_imlabs(placeholders, labels)

tf_fn_enqueue_op = get_fn_enqueue_op(placeholders, labels, idx_placeholders, tf_q_grp)

def _worker(n_epochs, queue, sess):
    """
    Worker process
        n_epochs: The number of epochs to run for 
        tf_fn_q: the filename queue 
        tf_lab_q: the label queue 
        sess: a tensorflow session
    """
    enq_c = 0
    idx_l = range(data.shape[0])
    cur_batch_ims = []
    cur_labels = []
    cur_idxs = []
    for i in range(n_epochs+1):
        print 'Starting epoch %i' % i
        # generate a shuffle
        np.random.shuffle(idx_l)
        cur_sidx = 0
        for idx in idx_l:
            l1 = np.zeros(BATCH_SIZE)
            l1_idx = cur_sidx
            cur_sidx += 1
            l2 = np.zeros(BATCH_SIZE)
            l2_idx = cur_sidx
            cur_sidx += 1
            cur_sidx = cur_sidx % BATCH_SIZE
            f1 = fnmap[data[idx, 0]]
            f2 = fnmap[data[idx, 1]]
            l1[l2_idx] = data[idx, 2]
            l2[l1_idx] = data[idx, 3]
            cur_batch_ims.append(f1)
            cur_batch_ims.append(f2)
            cur_labels.append(l1)
            cur_labels.append(l2)
            cur_idxs.append(data[idx, 0])
            cur_idxs.append(data[idx, 1])
            # try:
            #     # sess.run(tf_fn_enqueue_op, feed_dict={tf_fn_phd: f1})
            #     sess.run(tf_lab_enqueue_op, feed_dict={tf_lab_phd: l1.astype(np.float32)})
            #     # sess.run(tf_fn_enqueue_op, feed_dict={tf_fn_phd: f2})
            #     sess.run(tf_lab_enqueue_op, feed_dict={tf_lab_phd: l2.astype(np.float32)})
            # except:
            #     print 'ERROR!'
            #     return
            if len(cur_batch_ims) == BATCH_SIZE:
                enq_c += BATCH_SIZE
                print cur_idxs
                feed_dict = {x: y for x, y in zip(placeholders, cur_batch_ims)}
                for x, y in zip(labels, cur_labels):
                    feed_dict[x] = y
                for x, y in zip(idx_placeholders, cur_idxs):
                    feed_dict[x] = [y]
                sess.run(tf_fn_enqueue_op, feed_dict = feed_dict)
                cur_batch_ims = []
                cur_labels = []
                cur_idxs = []
                # print 'stuffs', stuffs1.eval(feed_dict = {x: y for x, y in zip(placeholders + labels, cur_batch_ims + cur_labels)})
                # stuffs = sess.run([stuffs1, stuffs2], feed_dict = {x: y for x, y in zip(placeholders + labels, cur_batch_ims + cur_labels)})
                return
    print 'Queue is done'

sess = tf.InteractiveSession()
blah = _worker(1, tf_q_grp, sess)
a, b, c = tf_q_grp.dequeue_many(BATCH_SIZE)
a_, b_, c_ = sess.run([a, b, c])

a_tf = np.squeeze(a_)
b_tf = np.squeeze(b_).astype(int)
c_tf = np.squeeze(c_).astype(int)

# now, assemble the ground truth to make sure it's accurate
big_mat = np.zeros((np.max(np.max(data,0)[:2])+1, np.max(np.max(data,0)[:2])+1))
for z1, z2, z3, z4 in data:
    big_mat[z1, z2] = z3
    big_mat[z2, z1] = z4
b_gt = np.zeros_like(b_tf)
idx1, idx2 = c_tf[np.arange(0, BATCH_SIZE, 2)], c_tf[np.arange(1, BATCH_SIZE, 2)]
# b_gt = big_mat[np.array([[x] for x in idx1]), idx2]
for n in np.arange(0, BATCH_SIZE, 2):
    ab = big_mat[c_tf[n], c_tf[n+1]]
    ba = big_mat[c_tf[n+1], c_tf[n]]
    b_gt[n, n+1] = ab
    b_gt[n+1, n] = ba

























