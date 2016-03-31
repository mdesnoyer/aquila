"""
Implements the loss function for training as well as the accuracy function.
"""
import tensorflow as tf


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
    return loss_


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
    pos_y_diff = tf.greater(y_diff_, zeros_)
    num_corr_ = tf.reduce_sum(tf.mul(pos_y_diff, m_))
    accuracy_ = num_corr_ / tf.reduce_sum(m_)
    return accuracy_
