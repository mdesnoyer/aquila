"""
Implements the RankNet loss function in TensorFlow given:
	y_ = the estimates of the values of y
	m_ = A matrix, where m_[i,j] = # of times i has beaten j
	p_ = A matrix of priors
"""

import numpy as np
import tensorflow as tf 

# oij = np.arange(-5, 5, .025)
# p = 0.75

# def c(oij, p):
# 	return -p * oij + np.log(1 + np.exp(oij))

# plot(oij, [c(x, p) for x in oij])

# # so p actually acts more like a regularizer, defining the minimum, i.e.,
# # when o_ij is:
# #
# # log( p_ij / (1 - p_ij) )


# let's try to implement the loss function in tensorflow
n_elem = 5
pairs = [list(np.random.choice(n_elem, 2, replace=False)) for x in range(n_elem * 3000)]  # the pairs to test
y = np.random.gamma(2, 0.5, n_elem)  # the ground truth

mult = np.zeros((n_elem, n_elem))
for i, j in pairs:
	# who won?
	yi = y[i]
	yj = y[j]
	if np.random.rand() < (yi / (yi + yj)):
		mult[i,j] += 1
	else:
		mult[j,i] += 1

# compute the loss with numpy
diff = y[None, :] - y[:, None]
p = np.ones_like(diff) * 0.75
loss = np.sum(mult * (-p * diff + np.log(1 + np.exp(diff)))) * 1./np.sum(mult)
print 'Numpy Loss:', loss

#sess = tf.InteractiveSession()

# instantiate the variables (here they're constants, but this isn't necessary)
y_ = tf.constant(y)  # y hat
m_ = tf.constant(mult)  # the multiplier
p_ = tf.constant(p)  # the priors


ones_ = tf.constant(np.ones_like(mult))  # a matrix of ones
y_m_ = tf.mul(y_, ones_)
y_diff_ = tf.sub(y_m_, tf.transpose(y_m_))
t_1_ = -tf.mul(p_, y_diff_)
t_2_ = tf.log(ones_ + tf.exp(y_diff_))
sum_ = tf.add(t_1_, t_2_)
mult_sum_ = tf.mul(m_, sum_)
loss_ = tf.reduce_sum(mult_sum_) / tf.reduce_sum(m_)
print 'Tensorflow Loss:', loss_.eval()
