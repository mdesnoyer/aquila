"""
Creates a dummy neural network, to test that the loss function I devised works.

Given two items, x1 and x2 where x1 was chosen x2, the neural net computes a
score for each: y_1, y_2. Then, the loss function is:

(for some eps, i.e., 1)

w = max(eps, | y_1 - y_2 |)^2
g = [y_1 > y_2]

loss = sqrt(sum(w * g))
"""
import numpy as np
import tensorflow as tf

n_items = 10  # the number of items over which to perform inference.
n_dims = 20  # the dimensionality of the dataset
n_samples = 1000  # the number of samples to draw

# randomly obtain ground truth
y = np.random.gamma(2, 0.5, size=n_items)
x = np.random.rand(n_items, n_dims)

# okay so idk what happened with the one below, so lets try this again
# y_hat = tf.Variable(
#     tf.truncated_normal([n_dims],
#                         stddev=1.0 / np.sqrt(n_dims)))



# run the simulation
outcome = []
for j in range(n_samples):
    c1, c2 = np.random.choice(n_items, 2, replace=False)
    if np.random.rand() < (y[c1] / (y[c1] + y[c2])):
        # then c1 wins
        outcome.append([c1, c2])
    else:
        # then c2 wins
        outcome.append([c2, c1])

data = np.array(outcome)



def data_y():
    idx = 0
    while True:
        yield x[data[idx, :]]
        idx += 1
        idx = idx % data.shape[0]

# tensorflow stuff
sess = tf.InteractiveSession()

X = tf.placeholder(tf.float32, [None, n_dims])
eps = tf.constant(0.01, dtype=tf.float32)
pow = tf.constant(2.0, dtype=tf.float32)
#X = tf.constant(tf.to_float(x[data[0,:]]))
with tf.name_scope('hidden1'):
    weights = tf.Variable(
        tf.truncated_normal([n_dims, n_dims],
                            stddev=1.0 / np.sqrt(n_dims)),
        name='weights')
    biases = tf.Variable(tf.zeros([n_dims]), name='biases')
    hidden1 = tf.nn.relu(tf.matmul(X, weights) + biases)

with tf.name_scope('hidden2'):
    weights = tf.Variable(
        tf.truncated_normal([n_dims, 1],
                            stddev=1.0 / np.sqrt(n_dims)),
        name='weights')
    biases = tf.Variable(tf.zeros([1]), name='biases')
    mmhidden = tf.matmul(hidden1, weights)
    hidden2 = tf.nn.relu(mmhidden + biases)

with tf.name_scope('loss'):
    tf_y_1, tf_y_2 = tf.split(0, 2, hidden2)
    gamma = tf.greater_equal(tf_y_2, tf_y_1)
    # omega = tf.pow(tf.maximum(eps, tf.abs(tf.sub(tf_y_1, tf_y_2))), pow)
    omega = tf.pow(tf.abs(tf.sub(tf_y_1, tf_y_2)), pow)
    loss = tf.sqrt(tf.to_float(gamma) * omega)

init = tf.initialize_all_variables()
tstep = tf.train.GradientDescentOptimizer(0.000001).minimize(loss)
dy = data_y()


sess.run(init)
for i in range(5):
    #j = sess.run([tstep, tf_y_1, tf_y_2, loss], feed_dict={X: dy.next()})
    #print '%.2f, %.2f, %.2f %i' % (j[1][0][0], j[2][0][0], j[3][0][0], i)
    q_ = sess.run([tstep, weights, loss], feed_dict={X: dy.next()})
    a_ = '[%.2f] ' % float(q_[2].flatten())
    b_ = ' '.join(['%5.2f' % x_ for x_ in q_[1].flatten()])
    print a_ + b_
# sess.run([tf_y_1, tf_y_2, loss], feed_dict={X: x[data[0, :], :]})
