"""
This is the second iteration of the loss function, which is designed to solve
BTL directly.

Woo! The linear, matrix, and tensorflow all work! yayy!

Note:
    ** The tensorflow loss expects two things:
        - W, a matrix of observed wins
        - y_hat, an estimate of the values of the values of y
    ** In production, we will exponentiate y, so that it's gaussian
    distributed.

"""

import numpy as np
import tensorflow as tf

n_elem = 5  # the number of items to simulate
n_trials_per = 10  # the number of trials per element
eps = 0.10  # the epsilon regularizer


# the ground truth
y = np.random.gamma(2, 0.5, n_elem)
# the estimate
y_hat = y + np.random.randn(n_elem)/5

# make a synthetic win matrix
W = y / (y[:,None] + y[None, :])
W[np.diag_indices(n_elem)] = 0
W = (W*n_trials_per).astype(int)


# matrix method ----------------------------------------------------
cnt = W + W.T
w_rat = (W + eps) / (W + W.T + 2*eps)
y_rat = y_hat / (y_hat + y_hat[:,None])
matr_error = (1. / np.sum(W)) * (np.sum(cnt * (w_rat - y_rat.T)**2))
# ------------------------------------------------------------------


# now, let's compute it linearly -----------------------------------
r = 1. / np.sum(W)
sm = 0
sMtx = np.zeros(W.shape)
yMtx = np.zeros(W.shape)
for i in range(n_elem):
    for j in range(n_elem):
        yi = y_hat[i]
        yj = y_hat[j]
        wij = W[i, j]
        wji = W[j, i]
        yth = yi / (yi + yj)
        q = ( (wij + eps) / (wij + wji + 2 * eps) - yth )**2
        sm += (wij + wji) * q
        sMtx[i, j] = q
        yMtx[i, j] = yth

lin_error = r * sm
# ------------------------------------------------------------------


# Tensorflow Method ------------------------------------------------
sess = tf.InteractiveSession()
W_ = tf.to_float(tf.constant(W))
eps_ = tf.to_float(tf.constant(eps))
y_hat_ = tf.to_float(tf.constant(y_hat))
ones_ = tf.to_float(tf.constant(np.ones(W.shape)))
y_rat_den_ = tf.mul(y_hat_, ones_)
W_rat_ = tf.div(W_ + eps, tf.add(W_, tf.transpose(W_)) + 2 * eps)
W_sm_mat_ = W_ + tf.transpose(W_)
y_rat_ = tf.div(y_hat_, tf.add(y_rat_den_, tf.transpose(y_rat_den_)))
inner_mat_ = W_sm_mat_ * tf.square(W_rat_ - tf.transpose(y_rat_))
matr_error_ = (1. / tf.reduce_sum(W_)) * tf.reduce_sum(inner_mat_)
# ------------------------------------------------------------------

print 'Matrix error: %.3f' % float(matr_error)
print 'Linear error: %.3f' % float(lin_error)
print 'Tensorflow error: %.3f' % float(matr_error_.eval())
