"""
Configuration for Aquila
"""
import os

# where to write event logs and checkpoints
train_dir = '/data/aquila_v2_snaps'
# the list of training data
if os.path.exists('/tmp/aquila_test_data/combined'):
    TRAIN_DATA = '/tmp/aquila_test_data/combined'
    TRAIN_IMAGES = '/tmp/aquila_test_data/images'
else:
    TRAIN_DATA = '/data/aquila_v2/combined'
    TRAIN_IMAGES = '/data/images'
SUBSET_SIZE = None  # if not None, will only train on 'SUBSET_SIZE' pairs.
TEST_DATA = None
TEST_IMAGES = None
NULL_IMAGE = '/tmp/null.jpg'
# 'train' or 'validation'
subset = 'train'

if subset == 'train':
    DATA_SOURCE = TRAIN_DATA
    IMAGE_SOURCE = TRAIN_IMAGES
else:
    DATA_SOURCE = TEST_DATA
    IMAGE_SOURCE = TEST_IMAGES

# Whether to log device placement.
log_device_placement = False  # this produces *so much* output!

# the number of preprocessing threads to create -- just 2 is more than
# sufficient, even for 4 gpus (apparently?)
num_preprocess_threads = 4
# how many gpus to use
num_gpus = 4


# the number of abstract features to learn
abs_feats = 1024

DEMOGRAPHIC_GROUPS = 9  # this really shouldn't be hardcoded, but meh.

LAPLACE_SMOOTHING_C = 0.05
# ---------------------------------------------------------------------------- #
# Flags governing the type of training.
# ---------------------------------------------------------------------------- #
# Whether or not to restore the logits.
restore_logits = True

# restore the pretrained model from this location
pretrained_model_checkpoint_path = ''  # '/data/aquila_snaps/model.ckpt-20000'

# the initial learning rate
initial_learning_rate = 0.5  # 0.05

# epochs after which learning rate decays
num_epochs_per_decay = 1  # 0.5 # within-epoch decay

# the learning rate decay factor
learning_rate_decay_factor = 0.999  # 0.65

# NOTE: Batch size should have the same parity as the
# average pair group.
BATCH_SIZE = 23

# are variable-length >.<
NUM_EPOCHS = 5000

# Constants dictating the learning rate schedule.
RMSPROP_DECAY = 0.9                # Decay term for RMSProp.
RMSPROP_MOMENTUM = 0.9             # Momentum in RMSProp.
RMSPROP_EPSILON = 1.0              # Epsilon term for RMSProp.

# regularization strength
WEIGHT_DECAY = 0  # 0.00004
