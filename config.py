"""
Configuration for Aquila
"""

# where to write event logs and checkpoints
train_dir = '/data/aquila_snaps_lowreg'

# 'train' or 'validation'
subset = 'train'

# how many gpus to use
num_gpus = 4

# Whether to log device placement.
log_device_placement = False  # this produces *so much* output!

# the number of preprocessing threads to create -- just 2 is more than
# sufficient, even for 4 gpus (apparently?) (actually no not anymore)
num_preprocess_threads = 3

# the number of abstract features to learn
abs_feats = 1024

# ---------------------------------------------------------------------------- #
# Flags governing the type of training.
# ---------------------------------------------------------------------------- #
# Whether or not to restore the logits.
restore_logits = True

# restore the pretrained model from this location
pretrained_model_checkpoint_path = ''  # '/data/aquila_snaps/model.ckpt-20000'

# the initial learning rate
initial_learning_rate = 0.05

# epochs after which learning rate decays
num_epochs_per_decay = 0.25  # within-epoch decay

# the learning rate decay factor
learning_rate_decay_factor = 0.65


BATCH_SIZE = 22

NUM_EPOCHS = 5

# Constants dictating the learning rate schedule.
RMSPROP_DECAY = 0.9                # Decay term for RMSProp.
RMSPROP_MOMENTUM = 0.9             # Momentum in RMSProp.
RMSPROP_EPSILON = 1.0              # Epsilon term for RMSProp.

# regularization strength
WEIGHT_DECAY = 1e-8  # 0.00004
DROPOUT_KEEP_PROB = 0.  # the dropout keep probability
