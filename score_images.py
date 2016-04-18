'''
This will extract the abstract features from all of our
training images to see how they scored.
'''

import tensorflow as tf 
from glob import glob
import os
from aquila.net import aquila_model as aquila
from aquila.net.slim import slim
from aquila.config import *


IMAGE_DIR = '/data/images/'
AQUILA_SNAP = '/data/aquila_snaps_lowreg/model.ckpt-250000'


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

_logits, _abst_feats = sess.run([logits, abst_feats], 
								feed_dict={fn_phd: images[0]})