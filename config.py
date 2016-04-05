"""
Configuration for Aquila
"""

# where to write event logs and checkpoints
train_dir = '/data/aquila_snaps'

# 'train' or 'validation'
subset = 'train'

# how many gpus to use
num_gpus = 4

# Whether to log device placement.
log_device_placement = False

# the number of preprocessing threads to create
num_preprocess_threads = 8

# the number of abstract features to learn
abs_feats = 1024

# ---------------------------------------------------------------------------- #
# Flags governing the type of training.
# ---------------------------------------------------------------------------- #
# Whether or not to restore the logits.
restore_logits = False

# restore the pretrained model from this location
pretrained_model_checkpoint_path = '/data/pretrained/model.ckpt-157585'

# the initial learning rate
initial_learning_rate = 0.05

# epochs after which learning rate decays
num_epochs_per_decay = 30.0

# the learning rate decay factor
learning_rate_decay_factor = 0.16


BATCH_SIZE = 32

NUM_EPOCHS = 200

# Constants dictating the learning rate schedule.
RMSPROP_DECAY = 0.9                # Decay term for RMSProp.
RMSPROP_MOMENTUM = 0.9             # Momentum in RMSProp.
RMSPROP_EPSILON = 1.0              # Epsilon term for RMSProp.