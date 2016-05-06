'''
This will extract the abstract features from all of our
training images to see how they scored.
'''

import sys
sos.path.insert(0, '/data')
import tensorflow as tf 
from glob import glob
import os
from aquila.net import aquila_model as aquila
from aquila.net.slim import slim
from aquila.config import *
import numpy as np 


IMAGE_DIR = '/data/images/'
AQUILA_SNAP = '/data/aquila_snaps_lowreg/model.ckpt-250000'
DEST = '/data/bestworst'

class Ranker():
	def __init__(self, n):
		'''
		Will store the top and bottom n given a key and a value.
		'''
		self.n = n
		self.top = [(None, -np.inf) for _ in range(n)]
		self.bottom = [(None, np.inf) for _ in range(n)]
		self.min_top = -np.inf
		self.max_bot = np.inf

	def insert(self, key, value):
		if value > self.min_top:
			self.top.append([key, value])
			self.top = sorted(self.top, key=lambda x: -x[1])[:self.n]
			self.min_top = self.top[-1][1]
		if value < self.max_bot:
			self.bottom.append([key, value])
			self.bottom = sorted(self.bottom, key=lambda x: -x[1])[1:]
			self.max_bot = self.bottom[0][1]


# fetch all the images
images = glob(os.path.join(IMAGE_DIR, '*.jpg'))


fn_phd = tf.placeholder(tf.string, shape=[])
raw_im = tf.read_file(fn_phd)
image = tf.image.decode_jpeg(raw_im, channels=3)
image = tf.expand_dims(image, 0)
image = tf.image.resize_bilinear(image, [299, 299],
                                 align_corners=False)

# instantiate aquila
with tf.device('/gpu:0'):
	logits, abst_feats = aquila.abst_feats(
		image, abs_feats, for_training=False,
	    restore_logits=True, scope=None,
	    regularization_strength=WEIGHT_DECAY,
	    dropout_keep_prob=DROPOUT_KEEP_PROB)

init = tf.initialize_all_variables()
sess = tf.InteractiveSession()
sess.run(init)

variables_to_restore = tf.get_collection(
                slim.variables.VARIABLES_TO_RESTORE)
restorer = tf.train.Saver(variables_to_restore)

# restore it!
restorer.restore(sess, AQUILA_SNAP)

# how will we store the best/worst of the images for each 
# feature?
# bw_logits = Ranker(10)
# abs_bestworst = dict()
# all_scores = []
# for i in range(abs_feats):
# 	abs_bestworst[i] = Ranker(5)
# for im_num, image in enumerate(images):
# 	if not im_num % 100:
# 		print '%i / %i' % (im_num, len(images))
# 	_logits, _abst_feats = sess.run([logits, abst_feats], 
# 								feed_dict={fn_phd: image})
# 	_abst_feats = _abst_feats.squeeze()
# 	bw_logits.insert(image, float(_logits))
# 	for i in range(abs_feats):
# 		abs_bestworst[i].insert(image, _abst_feats[i])
# 	all_scores.append(_logits)

# with open('/home/ubuntu/all_scores', 'w') as f:
# 	f.write(str(all_scores))

# with open(os.path.join(DEST, 'logits_best'), 'w') as f:
# 	for k, v in bw_logits.top:
# 		f.write('%s %.4f\n' % (k, v))

# with open(os.path.join(DEST, 'logits_worst'), 'w') as f:
# 	for k, v in bw_logits.bottom:
# 		f.write('%s %.4f\n' % (k, v))


# for i in range(abs_feats):
# 	with open(os.path.join(DEST, 'abst_feat_%i_best' % i), 'w') as f:
# 		for k, v in abs_bestworst[i].top:
# 			f.write('%s %.4f\n' % (k, v))

# 	with open(os.path.join(DEST, 'abst_feat_%i_worst' % i), 'w') as f:
# 		for k, v in abs_bestworst[i].bottom:
# 			f.write('%s %.4f\n' % (k, v))

bw_logits = Ranker(10)
abs_bestworst = dict()
all_scores = []
for i in range(abs_feats):
	abs_bestworst[i] = Ranker(5)
for im_num, image in enumerate(images):
	if not im_num % 100:
		print '%i / %i' % (im_num, len(images))
	_logits = sess.run([logits], feed_dict={fn_phd: image})
	with open('/home/ubuntu/all_scores_text', 'a') as f:
		f.write(str(_logits) + '\n')
	all_scores.append(_logits)

with open('/home/ubuntu/all_scores', 'w') as f:
	f.write(str(all_scores))