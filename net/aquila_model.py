"""
This mimics the behavior of the inception model definition. The actual net is
defined below, in 'slim.' This module imports the model and defines it
appropriately for a single tower (i.e., a GPU).
"""

from __future__ import absolute_import
from __future__ import division

import re
import tensorflow as tf
from net.slim import slim

FLAGS = tf.app.flags.FLAGS


# If a model is trained using multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

# ---------------------------------------------------------------------------- #
#                 These parameters are taken directly from inception
# ---------------------------------------------------------------------------- #
# Batch normalization. Constant governing the exponential moving average of
# the 'global' mean and variance for all activations.
BATCHNORM_MOVING_AVERAGE_DECAY = 0.9997

# The decay to use for the moving average.
MOVING_AVERAGE_DECAY = 0.9999


def inference(inputs, abs_feats=1024, for_training=True,
              restore_from_inc=True, scope=None):
    """
    Exports an inference op, along with the logits required for loss
    computation.

    :param inputs: An N x 299 x 299 x 3 sized float32 tensor (images)
    :param abs_feats: The number of abstract features to learn.
    :param for_training: Boolean, whether or not training is being performed.
    :param restore_from_inc: Restore for a pretrained inception model.
    :param scope: The name of the tower (i.e., GPU) this is being done on.
    :return: Logits, aux logits.
    """
    # Parameters for BatchNorm.
    batch_norm_params = {
        # Decay for the moving averages.
        'decay': BATCHNORM_MOVING_AVERAGE_DECAY,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
    }


