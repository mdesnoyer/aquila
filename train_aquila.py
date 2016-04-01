"""
Actually accomplishes the training of Aquila.
"""

from training.input import InputManager
from net.slim.aquila_model import aquila
from net.slim.losses import ranknet_loss
from net.slim.losses import accuracy
from scipy import io
