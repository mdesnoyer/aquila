"""
Implements the loss function for training as well as the accuracy function.
"""
import tensorflow as tf


LOSSES_COLLECTION = '_losses'


def l1_loss(tensor, weight=1.0, scope=None):
  """Define a L1Loss, useful for regularize, i.e. lasso.

  Args:
    tensor: tensor to regularize.
    weight: scale the loss by this factor.
    scope: Optional scope for op_scope.

  Returns:
    the L1 loss op.
  """
  with tf.op_scope([tensor], scope, 'L1Loss'):
    weight = tf.convert_to_tensor(weight,
                                  dtype=tensor.dtype.base_dtype,
                                  name='loss_weight')
    loss = tf.mul(weight, tf.reduce_sum(tf.abs(tensor)), name='value')
    tf.add_to_collection(LOSSES_COLLECTION, loss)
    return loss


def l2_loss(tensor, weight=1.0, scope=None):
  """Define a L2Loss, useful for regularize, i.e. weight decay.

  Args:
    tensor: tensor to regularize.
    weight: an optional weight to modulate the loss.
    scope: Optional scope for op_scope.

  Returns:
    the L2 loss op.
  """
  with tf.op_scope([tensor], scope, 'L2Loss'):
    weight = tf.convert_to_tensor(weight,
                                  dtype=tensor.dtype.base_dtype,
                                  name='loss_weight')
    loss = tf.mul(weight, tf.nn.l2_loss(tensor), name='value')
    tf.add_to_collection(LOSSES_COLLECTION, loss)
    return loss


def cross_entropy_loss(logits, one_hot_labels, label_smoothing=0,
                       weight=1.0, scope=None):
  """
  Define a Cross Entropy loss using softmax_cross_entropy_with_logits.

  It can scale the loss by weight factor, and smooth the labels.

  Args:
    logits: [batch_size, num_classes] logits outputs of the network .
    one_hot_labels: [batch_size, num_classes] target one_hot_encoded labels.
    label_smoothing: if greater than 0 then smooth the labels.
    weight: scale the loss by this factor.
    scope: Optional scope for op_scope.

  Returns:
    A tensor with the softmax_cross_entropy loss.
  """
  logits.get_shape().assert_is_compatible_with(one_hot_labels.get_shape())
  with tf.op_scope([logits, one_hot_labels], scope, 'CrossEntropyLoss'):
    num_classes = one_hot_labels.get_shape()[-1].value
    one_hot_labels = tf.cast(one_hot_labels, logits.dtype)
    if label_smoothing > 0:
      smooth_positives = 1.0 - label_smoothing
      smooth_negatives = label_smoothing / num_classes
      one_hot_labels = one_hot_labels * smooth_positives + smooth_negatives
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits,
                                                            one_hot_labels,
                                                            name='xentropy')
    weight = tf.convert_to_tensor(weight,
                                  dtype=logits.dtype.base_dtype,
                                  name='loss_weight')
    loss = tf.mul(weight, tf.reduce_mean(cross_entropy), name='value')
    tf.add_to_collection(LOSSES_COLLECTION, loss)
    return loss


def ranknet_loss(y, m_, conf=0.999, weight=1.0, scope=None):
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
    :param conf: "Confidence" (name is legacy from the original paper; it is
    perhaps more appropriate to call it "regularization". This effectively
    imposes an optimal separation between scores when there is precisely one
    win.
    :param weight: How much weight to assign to this function.
    :param scope: The scope for this operation.
    :return: The TensorFlow loss operation.
    """
    with tf.op_scope([y, m_], scope, 'RankNetLoss'):
        ones_ = tf.ones_like(m_, dtype=tf.float32)
        y_m_ = tf.mul(y, ones_)
        y_diff_ = tf.sub(y_m_, tf.transpose(y_m_))
        t_1_ = -tf.mul(conf*ones_, y_diff_)
        t_2_ = tf.log(ones_ + tf.exp(y_diff_))
        sum_ = tf.add(t_1_, t_2_)
        mult_sum_ = tf.mul(m_, sum_)
        loss_ = tf.reduce_sum(mult_sum_) / tf.reduce_sum(m_)
        tf.add_to_collection(LOSSES_COLLECTION, loss_)
        return weight * loss_


def ranknet_loss_demo(y, w, co, weight=1.0, scope=None):
    """
    Implements the ranknet loss with demographic personalization.

    Notes: Each image has N features and there are M demographic 'bins'. Assume
    that there are B images per batch.

    :param y: The B x D predictions for this image.
    :param w: The normalized win matrix, of shape B x B x D
    :param co: The confidence matrix, of shape B x B x D
    :param weight: The relative weighting of the loss.
    :param scope: The scope for this operation.

    :return: The TensorFlow loss operation.
    """
    with tf.op_scope([y, w, co], scope, 'RankNetLossD'):
        # compute the score difference
        Sd_t1 = tf.expand_dims(y, 1)
        Sd_t2 = tf.expand_dims(y, 0)
        dS = Sd_t1 - Sd_t2
        Wd = w + tf.transpose(w, perm=[1, 0, 2])
        Wd = tf.clip_by_value(Wd, 1, 10**8)
        Wn = tf.div(w, Wd)  # the win ratios
        t_1= -tf.mul(co, dS)
        t_2 = tf.log(1 + tf.exp(dS))
        loss = tf.reduce_sum(tf.mul((t_1 + t_2), Wn)) / tf.reduce_sum(w)
        tf.add_to_collection(LOSSES_COLLECTION, weight * loss)
        return weight * loss


def accuracy(y, m_, scope=None):
    """
    Computes accuracy (fraction of correct predictions).

    NOTES:
        y and m_ must be the same shape!

    :param y: A tensor of predictions from the network. (float32)
    :param m_: The win matrix, m_[i,j] = number of times i has beaten j. (
    float32)
    :param scope: The operation scope.
    :return: The TensorFlow accuracy operation.
    """
    with tf.op_scope([y, m_], scope, 'Accuracy'):
        ones_ = tf.ones_like(m_, dtype=tf.float32)
        zeros_ = tf.zeros_like(m_, dtype=tf.float32)
        y_m_ = tf.mul(y, ones_)
        y_diff_ = tf.sub(y_m_, tf.transpose(y_m_))
        pos_y_diff = tf.to_float(tf.greater(y_diff_, zeros_))
        num_corr_ = tf.reduce_sum(tf.mul(pos_y_diff, m_))
        accuracy_ = num_corr_ / tf.reduce_sum(m_)
        return accuracy_


def accuracy_demo(y, w, scope=None):
    """
    Computes accuracy (fraction of correct predictions).

    :param y: A tensor of predictions from the network. (float32)
    :param w: The win matrix, m_[i,j,k] = number of times i has beaten j as
    decided by demographic group k. (B x B x D)
    :param scope: The operation scope.
    :return: The TensorFlow accuracy operation.
    """
    with tf.op_scope([y, w], scope, 'AccuracyD'):
        # compute the score difference
        Sd_t1 = tf.expand_dims(y, 1)
        Sd_t2 = tf.expand_dims(y, 0)
        dS = Sd_t1 - Sd_t2
        zeros_ = tf.zeros_like(dS, dtype=tf.float32)
        pos_y_diff = tf.to_float(tf.greater(dS, zeros_))
        num_corr_ = tf.reduce_sum(tf.mul(pos_y_diff, w))
        accuracy_ = num_corr_ / tf.reduce_sum(w)
        return accuracy_