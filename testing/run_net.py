'''
This will generate and train the network. Mainly, this is to test the input
and loss functions.
'''
import numpy as np
import tensorflow as tf 
import threading
from glob import glob

N_EPOCHS = 20000
BATCH_SIZE = 16
IM_H = 128
IM_W = 128
LEARNING_RATE = 0.01
DISPLAY_STEP = 200
N_INPUT = IM_H * IM_W
STD_DEV = 0.01
WLFN = '/Users/davidlea/Desktop/testing/win_data/winlist.npy'
MPFN = '/Users/davidlea/Desktop/testing/win_data/id_2_fn'


tf_fn_phd = tf.placeholder(tf.string, shape=[])
tf_lab_phd = tf.placeholder(tf.int32, shape=[BATCH_SIZE])


def read_images_in(tf_fn_phd):
    """
    Takes a raw filename and enqueues the file specified

    fn: a placeholder
    """
    raw_im_ = tf.read_file(tf_fn_phd)
    im_ = tf.image.decode_jpeg(raw_im_)
    r_im_ = tf.image.resize_images(im_, IM_H, IM_W)
    return r_im_


tf_fn_q = tf.FIFOQueue(128, [tf.float32], shapes=[[IM_H, IM_W, 1]])
tf_lab_q = tf.FIFOQueue(128, [tf.float32], shapes=[[BATCH_SIZE]])


def get_fn_enqueue_op(tf_fn_phd):
    """
    Takes a raw filename and enqueues the file specified

    fn: a placeholder
    """
    raw_im_ = tf.read_file(tf_fn_phd)
    im_ = tf.image.decode_jpeg(raw_im_)
    r_im_ = tf.image.resize_images(im_, IM_H, IM_W)
    enq_op = tf_fn_q.enqueue([tf.to_float(r_im_)])
    return enq_op

# read all the files in for validation
ims = glob('/Users/davidlea/Desktop/testing/images/*.jpg')
ims = np.sort(ims)
tf_ims = []
im_reader_op = read_images_in(tf_fn_phd)
for im in ims:
    tf_ims.append(read_images_in(tf.constant(im)))
val_X = tf.pack(tf_ims)
    

# tf_fn_enqueue_op = tf_fn_q.enqueue([tf_fn_phd])
tf_fn_enqueue_op = get_fn_enqueue_op(tf_fn_phd)
tf_lab_enqueue_op = tf_lab_q.enqueue([tf.to_float(tf_lab_phd)])

data = np.load(WLFN)
fnmap = dict()
with open(MPFN, 'r') as f:
    for line in f:
        idx, fn = line.strip().split(',')
        fnmap[int(idx)] = fn


def _worker(n_epochs, tf_fn_Q, tf_lab_Q, sess):
    """
    Worker process
        n_epochs: The number of epochs to run for
        tf_fn_q: the filename queue
        tf_lab_q: the label queue
        sess: a tensorflow session
    """
    enq_c = 0
    idx_l = range(data.shape[0])
    try:
        cur_sidx = 0
        for i in range(n_epochs+1):
            print 'Starting epoch %i' % i
            # generate a shuffle
            np.random.shuffle(idx_l)
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
                try:
                    sess.run(tf_fn_enqueue_op, feed_dict={tf_fn_phd: f1})
                    sess.run(tf_lab_enqueue_op, feed_dict={tf_lab_phd: l1.astype(np.float32)})
                    sess.run(tf_fn_enqueue_op, feed_dict={tf_fn_phd: f2})
                    sess.run(tf_lab_enqueue_op, feed_dict={tf_lab_phd: l2.astype(np.float32)})
                except:
                    print 'ERROR!'
                    return
                enq_c += 1
    except Exception as e:
        print 'GERRORRRR', e.message
    print 'Queue is done'


# tensorflow crap
x = tf.placeholder(tf.float32, [BATCH_SIZE, N_INPUT])
y = tf.placeholder(tf.float32, [BATCH_SIZE, BATCH_SIZE])
ones_ = tf.constant(np.ones((BATCH_SIZE, BATCH_SIZE)).astype(np.float32))  # a matrix of ones


# Create AlexNet model
def conv2d(name, l_input, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, 1, 1, 1], padding='SAME'),b), name=name)


def max_pool(name, l_input, k):
    return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)


def norm(name, l_input, lsize=4):
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)


def alex_net(inp_queue, _weights, _biases):
    # Reshape input picture
    _X = inp_queue.dequeue_many(BATCH_SIZE)
    tf.image_summary('inputs', _X, max_images=3, collections=None, name=None)
    # Convolution Layer
    conv1 = conv2d('conv1', _X, _weights['wc1'], _biases['bc1'])
    print conv1.get_shape()
    # Max Pooling (down-sampling)
    pool1 = max_pool('pool1', conv1, k=4)
    print pool1.get_shape()
    # Apply Normalization
    norm1 = norm('norm1', pool1, lsize=4)
    print norm1.get_shape()
    # Convolution Layer
    conv2 = conv2d('conv2', norm1, _weights['wc2'], _biases['bc2'])
    print conv2.get_shape()
    # Max Pooling (down-sampling)
    pool2 = max_pool('pool2', conv2, k=4)
    print pool2.get_shape()
    # Apply Normalization
    norm2 = norm('norm2', pool2, lsize=4)
    print norm2.get_shape()

    # Convolution Layer
    conv3 = conv2d('conv3', norm2, _weights['wc3'], _biases['bc3'])
    print conv3.get_shape()
    # Max Pooling (down-sampling)
    pool3 = max_pool('pool3', conv3, k=2)
    print pool3.get_shape()
    # Apply Normalization
    norm3 = norm('norm3', pool3, lsize=4)
    print norm3.get_shape()

    # Fully connected layer
    dense1 = tf.reshape(norm3, [-1, _weights['wd1'].get_shape().as_list()[0]]) # Reshape conv3 output to fit dense layer input
    print dense1.get_shape()
    dense1 = tf.nn.relu(tf.matmul(dense1, _weights['wd1']) + _biases['bd1'], name='fc1') # Relu activation
    print dense1.get_shape()
    dense2 = tf.nn.relu(tf.matmul(dense1, _weights['wd2']) + _biases['bd2'], name='fc2') # Relu activation
    print dense2.get_shape()
    # Output, class prediction
    out = tf.matmul(dense2, _weights['out']) + _biases['out']
    return out, _X

def alex_net_val(X_, _weights, _biases):
    # Convolution Layer
    conv1 = conv2d('conv1', X_, _weights['wc1'], _biases['bc1'])
    # Max Pooling (down-sampling)
    pool1 = max_pool('pool1', conv1, k=4)
    # Apply Normalization
    norm1 = norm('norm1', pool1, lsize=4)
    # Convolution Layer
    conv2 = conv2d('conv2', norm1, _weights['wc2'], _biases['bc2'])
    # Max Pooling (down-sampling)
    pool2 = max_pool('pool2', conv2, k=4)
    # Apply Normalization
    norm2 = norm('norm2', pool2, lsize=4)

    # Convolution Layer
    conv3 = conv2d('conv3', norm2, _weights['wc3'], _biases['bc3'])
    # Max Pooling (down-sampling)
    pool3 = max_pool('pool3', conv3, k=2)
    # Apply Normalization
    norm3 = norm('norm3', pool3, lsize=4)

    # Fully connected layer
    dense1 = tf.reshape(norm3, [-1, _weights['wd1'].get_shape().as_list()[0]]) # Reshape conv3 output to fit dense layer input
    dense1 = tf.nn.relu(tf.matmul(dense1, _weights['wd1']) + _biases['bd1'], name='fc1') # Relu activation
    dense2 = tf.nn.relu(tf.matmul(dense1, _weights['wd2']) + _biases['bd2'], name='fc2') # Relu activation
    # Output, class prediction
    out = tf.matmul(dense2, _weights['out']) + _biases['out']
    return out


   # Store layers weight & bias
weights = {
    'wc1': tf.Variable(tf.random_normal([3, 3, 1, 64], stddev=STD_DEV)),
    'wc2': tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=STD_DEV)),
    'wc3': tf.Variable(tf.random_normal([3, 3, 128, 256], stddev=STD_DEV)),
    'wd1': tf.Variable(tf.random_normal([4*4*256, 1024], stddev=STD_DEV)),
    'wd2': tf.Variable(tf.random_normal([1024, 1024], stddev=STD_DEV)),
    'out': tf.Variable(tf.random_normal([1024, 1], stddev=STD_DEV))
}
biases = {
    'bc1': tf.Variable(tf.random_normal([64], stddev=STD_DEV)),
    'bc2': tf.Variable(tf.random_normal([128], stddev=STD_DEV)),
    'bc3': tf.Variable(tf.random_normal([256], stddev=STD_DEV)),
    'bd1': tf.Variable(tf.random_normal([1024], stddev=STD_DEV)),
    'bd2': tf.Variable(tf.random_normal([1024], stddev=STD_DEV)),
    'out': tf.Variable(tf.random_normal([1], stddev=STD_DEV))
}

def get_loss(y_):
    """
    Returns a loss object given the output of a net and the label
    """
    m_ = tf_lab_q.dequeue_many(BATCH_SIZE)
    y_m_ = tf.mul(y_, ones_)
    y_diff_ = tf.sub(y_m_, tf.transpose(y_m_))
    t_1_ = -tf.mul(0.95*ones_, y_diff_)
    t_2_ = tf.log(ones_ + tf.exp(y_diff_))
    sum_ = tf.add(t_1_, t_2_)
    mult_sum_ = tf.mul(m_, sum_)
    loss_ = tf.reduce_sum(mult_sum_) / tf.reduce_sum(m_)
    return loss_, m_

def ranknet_loss(y, m_):
    """
    Implements the RankNet loss function given the outputs of a net and the
    outcome matrix. Also returns the accuracy (fraction of correct predictions).

    NOTES:
        y and m_ must be the same shape!

        This doesnt do the ridiculous loss collection crap that the other
        implementation does. It's vastly too complex for our purposes.

    :param y: A tensor of predictions from the network. (float32)
    :param m_: The win matrix, m_[i,j] = number of times i has beaten j. (
    float32)
    :return: The TensorFlow loss operation.
    """
    conf = 1.0
    ones_ = tf.ones_like(m_, dtype=tf.float32)
    y_m_ = tf.mul(y, ones_)
    y_diff_ = tf.sub(y_m_, tf.transpose(y_m_))
    t_1_ = -tf.mul(conf*ones_, y_diff_)
    t_2_ = tf.log(ones_ + tf.exp(y_diff_))
    sum_ = tf.add(t_1_, t_2_)
    mult_sum_ = tf.mul(m_, sum_)
    loss_ = tf.reduce_sum(mult_sum_) / tf.reduce_sum(m_)
    return loss_, m_


def accuracy(y, m_):
    """
    Computes accuracy (fraction of correct predictions).

    NOTES:
        y and m_ must be the same shape!

    :param y: A tensor of predictions from the network. (float32)
    :param m_: The win matrix, m_[i,j] = number of times i has beaten j. (
    float32)
    :return: The TensorFlow accuracy operation.
    """
    ones_ = tf.ones_like(m_, dtype=tf.float32)
    zeros_ = tf.zeros_like(m_, dtype=tf.float32)
    y_m_ = tf.mul(y, ones_)
    y_diff_ = tf.sub(y_m_, tf.transpose(y_m_))
    pos_y_diff = tf.to_float(tf.greater(y_diff_, zeros_))
    num_corr_ = tf.reduce_sum(tf.mul(pos_y_diff, m_))
    accuracy_ = num_corr_ / tf.reduce_sum(m_)
    return accuracy_

# Construct model
pred, inp = alex_net(tf_fn_q, weights, biases)
#cost, m_mat = get_loss(pred)
m_ = tf_lab_q.dequeue_many(BATCH_SIZE)
m_4d_ = tf.reshape(m_, [1, BATCH_SIZE, BATCH_SIZE, 1])
tf.image_summary('win_matrix', m_4d_, max_images=1, collections=None, name=None)

cost, m_mat = ranknet_loss(pred, m_)
acc = accuracy(pred, m_)
accuracy_summary = tf.scalar_summary("accuracy", acc)
cost_summary = tf.scalar_summary("cost", cost)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

init = tf.initialize_all_variables()

val_pred = alex_net_val(val_X, weights, biases)

# determine the number of training iterations
n_batches_per_epoch = data.shape[0] / BATCH_SIZE
n_iterations = n_batches_per_epoch * N_EPOCHS

merged = tf.merge_all_summaries()

def_graph = tf.get_default_graph()
ops = def_graph.get_operations()
ops_dict = {cop.name: cop for cop in ops}
# launch the graph
p = None


# p = plot(av_accs)
# p = p[0]
# ax = p.get_axes()
m_mats = []
with tf.Session() as sess:
    writer = tf.train.SummaryWriter("/tmp/toy", sess.graph_def)
    thread = threading.Thread(target=_worker, args=(N_EPOCHS, tf_fn_q, tf_lab_q, sess))
    thread.daemon = True
    thread.start()    
    sess.run(init)
    print val_X.get_shape()
    # there are 16 * 10 trials per epoch, so 10 batches per epoch
    for c_iter in range(n_iterations):
        #_loss, _opti, _acc, _m_mat = sess.run([cost, optimizer, acc, m_mat])
        # _loss, _opti = sess.run([cost, optimizer])
        summary_str, _loss, _opti, _acc, _m_mat = sess.run([merged, cost, optimizer, acc, m_mat])
        #m_mats.append(_m_mat)
        if not c_iter % 10:
            writer.add_summary(summary_str, c_iter)
        # if not c_iter % DISPLAY_STEP:
        #     all_pred = sess.run([val_pred, ])
        #     all_pred = np.squeeze(np.array(all_pred[0]))
        #     #print "All predictions:",all_pred
        #     all_pred -= np.min(all_pred)
        #     all_pred /= np.max(all_pred)
        #     try:
        #         p.set_ydata(all_pred)
        #     except:
        #         p = plot(all_pred)
        #         p = p[0]
        #     pause(0.1)
        # accs.append(_acc)
        # av_accs.append(np.mean(accs[-50:]))
        # if not c_iter % DISPLAY_STEP:
        #     p.set_data(np.arange(len(av_accs)), av_accs)
        #     ax.set_xlim([0, len(av_accs)])
        #     ax.set_ylim([0, 1])
        #     pause(0.1)
        # print "Iter", c_iter, "minibatch loss:", _loss, "minibatch accuracy:", _acc
        print "Iter", c_iter, "minibatch loss:", _loss, "minibatch accuracy:", _acc

# it was failing because the win matrix was not being generated correctly...














