"""
Uses aquila_train to train a model from the configuration specified in
config.py.

NOTES:
    Either my creation of the win matrices was buggy, or we have an issue
    with the lil matrices used to create the sparse matrices. In practice,
    it appears to be both. From now on, we won't be using sparse matrices,
    we're going to be reading an enumeration of all win events in the data.
"""

from PIL import Image
import numpy as np
from training.input import InputManager
from training.input import get_enqueue_op
from aquila_train import _tower_loss
from net import aquila_model as aquila
from config import *
import tensorflow as tf
from net import slim

x = np.random.rand(300, 300)
print x
np.set_printoptions(precision=2, linewidth=200, edgeitems=12,
                    suppress=False)
print x

# first, create a null image
imarray = np.random.rand(314, 314, 3) * 255
im = Image.fromarray(imarray.astype('uint8')).convert('RGBA')
im.save(NULL_IMAGE)

static = False

if not static:
  tf_queue = tf.FIFOQueue(BATCH_SIZE * num_gpus * 2,
                          [tf.float32, tf.float32, tf.float32, tf.string],
                          shapes=[[299, 299, 3],
                                  [BATCH_SIZE, DEMOGRAPHIC_GROUPS-1],
                                  [BATCH_SIZE, DEMOGRAPHIC_GROUPS-1],
                                  []])
  image_phds = [tf.placeholder(tf.string, shape=[]) for _ in range(BATCH_SIZE)]
  label_phds = [tf.placeholder(tf.float32, shape=[BATCH_SIZE,
                                                  DEMOGRAPHIC_GROUPS-1])
                for _ in range(BATCH_SIZE)]
  conf_phds = [tf.placeholder(tf.float32, shape=[BATCH_SIZE,
                                                 DEMOGRAPHIC_GROUPS-1])
               for _ in range(BATCH_SIZE)]
  enqueue_op = get_enqueue_op(image_phds, label_phds, conf_phds, tf_queue)

  im = InputManager(image_phds, label_phds, conf_phds, tf_queue,
                    enqueue_op, num_epochs=NUM_EPOCHS,
                    num_qworkers=num_preprocess_threads)
else:
  inputs, labels, conf, filenames = np.load('/data/testv.npy')
  inputs = tf.constant(np.array([x for x in inputs]))
  labels = tf.constant(np.array([x for x in labels]))
  conf = tf.constant(np.array([x for x in conf]))

# tf_queue_t = tf.FIFOQueue(BATCH_SIZE * num_gpus * 2,
#                         [tf.float32, tf.float32, tf.float32, tf.string],
#                         shapes=[[299, 299, 3],
#                                 [BATCH_SIZE, DEMOGRAPHIC_GROUPS-1],
#                                 [BATCH_SIZE, DEMOGRAPHIC_GROUPS-1],
#                                 []])
# image_phds_t = [tf.placeholder(tf.string, shape=[]) for _ in range(BATCH_SIZE)]
# label_phds_t = [tf.placeholder(tf.float32, shape=[BATCH_SIZE,
#                                                  DEMOGRAPHIC_GROUPS-1])
#                 for _ in range(BATCH_SIZE)]
# conf_phds_t = [tf.placeholder(tf.float32, shape=[BATCH_SIZE,
#                                                  DEMOGRAPHIC_GROUPS-1])
#                for _ in range(BATCH_SIZE)]
# enqueue_op_t = get_enqueue_op(image_phds_t, label_phds_t, conf_phds_t,
#                               tf_queue_t)

# val_im = InputManager(image_phds_t, label_phds_t, conf_phds_t, tf_queue_t,
#                       enqueue_op_t, num_epochs=999999,
#                       num_qworkers=1, data_source=TEST_DATA, is_training=False)


global_step = tf.get_variable(
                'global_step', [],
                initializer=tf.constant_initializer(0), trainable=False)

lr = 1e-7
WEIGHT_DECAY = 0.000001
# opt = tf.train.RMSPropOptimizer(lr, RMSPROP_DECAY,
#                                     momentum=RMSPROP_MOMENTUM,
#                                     epsilon=RMSPROP_EPSILON)
# opt = tf.train.GradientDescentOptimizer(lr)
opt = tf.train.AdamOptimizer(lr)

with tf.name_scope('testing') as scope:
  if not static:
    inputs, labels, conf, filenames = im.tf_queue.dequeue_many(BATCH_SIZE)
  else:
    pass
  logits, aux_logits = aquila.inference(inputs, abs_feats, for_training=True,
                            restore_logits=restore_logits, scope=scope,
                            regularization_strength=WEIGHT_DECAY)
  logits = [logits, aux_logits]
  aquila.loss(logits, labels, conf)
  accuracy = aquila.accuracy(logits, labels)
  losses = tf.get_collection(slim.losses.LOSSES_COLLECTION, scope)
  regularization_losses = tf.get_collection(
                              tf.GraphKeys.REGULARIZATION_LOSSES)
  loss = tf.add_n(losses + regularization_losses, name='total_loss')
  batchnorm_updates = tf.get_collection(
                      slim.ops.UPDATE_OPS_COLLECTION, scope)
  grads = opt.compute_gradients(loss)
  pgrads = [x[0] for x in grads]
  grads = [[tf.clip_by_value(x[0], -0.5, 0.5), x[1]] for x in grads]
apply_gradient_op = opt.apply_gradients(grads, global_step=                     global_step)
variable_averages = tf.train.ExponentialMovingAverage(
            aquila.MOVING_AVERAGE_DECAY, global_step)

# Another possibility is to use tf.slim.get_variables().
variables_to_average = (tf.trainable_variables() +
                        tf.moving_average_variables())
variables_averages_op = variable_averages.apply(variables_to_average)
batchnorm_updates_op = tf.group(*batchnorm_updates)
train_op = tf.group(apply_gradient_op)

init = tf.initialize_all_variables()
sess = tf.InteractiveSession()
the_vars = [x[1] for x in grads]
the_grads = [x[0] for x in grads]

sess.run(init)

# we can't restore the variables for various reasons :-[

# pretrained_model_checkpoint_path = '/data/aquila_v2_snaps/model.ckpt-80000'
# from glob import glob
# files = glob('/data/aquila_v2_snaps/*')
# for pretrained_model_checkpoint_path in files:
#   try:
#     variables_to_restore = tf.get_collection(
#             slim.variables.VARIABLES_TO_RESTORE)
#     restorer = tf.train.Saver(variables_to_restore)
#     restorer.restore(sess, pretrained_model_checkpoint_path)
#     print 'yey'
#     break
#   except:
#     continue
if not static:
  im.start(sess)
# inputs, labels, conf, filenames = sess.run([inputs, labels, conf, filenames])
# np.save('/data/testv', [inputs, labels, conf, filenames])

nacc = float("inf")
initacc = 0.5
initloss = 1.0
coef = 0.999
# ngrads = sess.run(the_grads)
pinputs = None
plogits = None
cnt = 0

def run_op(requested, sess):
  ''' deals out the outputs for singletons / lists of lists '''
  inp = []
  out_map = []
  for n, i in enumerate(requested):
    if type(i) is not list and type(i) is not tuple:
      out_map.append(n)
      inp.append(i)
    else:
      for k in i:
        out_map.append(n)
        inp.append(k)
  out = sess.run(inp)
  aout = []
  clist = []
  cidx = 0
  for val, idx in zip(out, out_map):
    if idx == cidx:
      clist.append(val)
    else:
      if len(clist) > 1:
        aout.append(clist)
      else:
        aout.append(clist[0])
      clist = [val,]
      cidx = idx
  aout.append(clist)
  return aout

# while True:
losses_list = []
accuracy_list = []
grad_norms = []
var_norms = []
all_datas = []
initacc = 0.50
initloss = None
citer = 0
datas = []

'''
_, nloss, nacc, nlogits, nlabels, nthe_grads, nthe_vars = run_op([train_op, loss, accuracy, logits[0], labels, pgrads, the_vars], sess)
'''
while True:
  # nlogits, nlabels, ninputs = sess.run([logits[0], labels, inputs])
  # nacc, nlogits, nlabels, ninputs = sess.run([accuracy, logits[0], labels, inputs])
  _, nloss, nacc, nlogits, nlabels, nthe_grads, nthe_vars = run_op([train_op, loss, accuracy, logits[0], labels, pgrads, the_vars], sess)
  losses_list.append(nloss)
  accuracy_list.append(nacc)
  # grad_norms.append([np.linalg.norm(x) for x in nthe_grads])
  # var_norms.append([np.linalg.norm(x) for x in nthe_vars])
  max_grad = np.max([np.max(np.abs(x)) for x in nthe_grads])
  if initacc is None:
    initacc = nacc
  if initloss is None:
    initloss = nloss
  # _, nloss, nacc, nlogits, nlabels = sess.run([train_op, loss, accuracy, logits[0], labels])
  if np.isfinite(nloss):
    initloss = coef * initloss + (1 - coef) * nloss
  if np.isfinite(nacc):
    initacc = coef * initacc + (1 - coef) * nacc

  print '[%i] %.2g %.3f | %g %g | mg: %g' % (citer, nacc, initacc, nloss, initloss, max_grad)
  citer += 1
  datas.append([nacc, initacc, nloss, initloss, max_grad])
# print 'achieved 0 acc, acquiring end points'
# epk = end_points.keys()
# end_pts = sess.run([end_points[z] for z in epk])
# end_pts = {k:v for k, v in zip(epk, end_pts)}
# logits_n, acc_n, labels_n = sess.run([logits[0], accuracy, labels])
# grads_n = sess.run(the_grads)
# vars_n = sess.run(the_vars)
# v = sess.run(ap_g_o)
# #im.stop()
# # val_im.stop()
# #sess.close()

# # no batchnorm
# # INIT_STD  logit_norm
# # 0.001 1.33484e+06
# # 0.0001 10278.9
# # 1e-09 4.21724e-06
# # 1e-06 2.82275
# # 1e-07 0.0118356  # so, to achieve the logit norm seen with batch
# # normalization, we have to use an initial standard deviation of 1e-07



# # with batchnorm
# # 0.001 0.00508287
# # 1e-09 5.41281e-15

# # so the norm ~ 1000 / INIT_STD
# # print '%g %g' % (INIT_STD, np.linalg.norm(logits_n))

# # mean / max / std grad norm without batchnorm:
# # mean / max / std grad norm with batchnorm:

# av = [np.linalg.norm(x) for x in grads_n]
# print np.mean(av), np.max(av), np.std(av)

# av = [np.linalg.norm(x) for x in vars_n]
# print np.mean(av), np.max(av), np.std(av)
