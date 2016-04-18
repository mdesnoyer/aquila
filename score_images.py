'''
This will extract the abstract features from all of our
training images to see how they scored.
'''

import tensorflow as tf 
from glob import glob
import os
from aquila.net import aquila_model as aquila
from aquila.net.slim import slim
from config import *


IMAGE_DIR = '/data/images/'
AQUILA_SNAP = '/data/aquila_snaps_lowreg/model.ckpt-250000'


# fetch all the images
images = glob(os.path.join(IMAGE_DIR, '*.jpg'))


inputs = tf.placeholder(tf.float32, shape=[22, 299, 299, 3])
# instantiate aquila
logits, abst_feats = aquila.abst_feats(
	inputs, abs_feats, for_training=False,
    restore_logits=True, scope=scope,
    regularization_strength=WEIGHT_DECAY,
    dropout_keep_prob=DROPOUT_KEEP_PROB))