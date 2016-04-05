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
# ---------------------------------------------------------------------------- #


def inference(inputs, abs_feats=1024, for_training=True,
              restore_logits=True, scope=None):
    """
    Exports an inference op, along with the logits required for loss
    computation.

    :param inputs: An N x 299 x 299 x 3 sized float32 tensor (images)
    :param abs_feats: The number of abstract features to learn.
    :param for_training: Boolean, whether or not training is being performed.
    :param restore_logits: Restore the logits. This should only be done if the 
    model is being trained on a previous snapshot of Aquila. If training from
    scratch, or transfer learning from inception, this should be false as the
    number of abstract features will likely change.
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
    # Set weight_decay for weights in Conv and FC layers.
    with slim.arg_scope([slim.ops.conv2d, slim.ops.fc], weight_decay=0.000004):
        with slim.arg_scope([slim.ops.conv2d],
                stddev=0.1,
                activation=tf.nn.relu,
                batch_norm_params=batch_norm_params):
            # Force all Variables to reside on the CPU.
            with slim.arg_scope([slim.variables.variable], device='/cpu:0'):
                logits, endpoints = slim.aquila.aquila(
                    inputs,
                    dropout_keep_prob=0.8,
                    num_abs_features=abs_feats,
                    is_training=for_training,
                    restore_logits=restore_logits,
                    scope=scope)

    # Add summaries for viewing model statistics on TensorBoard.
    _activation_summaries(endpoints)

    # Grab the logits associated with the side head. Employed during training.
    auxiliary_logits = endpoints['aux_logits']

    return logits, auxiliary_logits


def loss(logits, labels):
    """
    Adds all losses for the model.

    Note the final loss is not returned. Instead, the list of losses are collected
    by slim.losses. The losses are accumulated later along with the regularization
    loss.

    :param logits: The predicted image scores as a list of [BATCH_SIZE] float32
    tensors.
    :param labels: The labels, a [BATCH_SIZE, BATCH_SIZE] float32 tensor.
    :returns: None.
    """
    slim.losses.ranknet_loss(logits[0], labels, weight=1.0)
    slim.losses.ranknet_loss(logits[1], labels, weight=0.4, scope='aux_loss')


def accuracy(logits, labels):
    """
    Computes the accuracy of the output of the final logit layer. We
    disregard the action of the auxiliary logit head in this case.

    :param logits:  The predicted image scores as a list of [BATCH_SIZE]
    float32 tensors. Note that only the first element of this list is used.
    :param labels: The labels, a [BATCH_SIZE, BATCH_SIZE] float32 tensor.
    :return: The accuracy op.
    """
    return slim.losses.accuracy(logits[0], labels)


def _activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.

    Args:
        x: Tensor
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.histogram_summary(tensor_name + '/activations', x)
    tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _activation_summaries(endpoints):
    with tf.name_scope('summaries'):
        for act in endpoints.values():
            _activation_summary(act)

    


