# Import utility functions from model_tests folder
import sys
sys.path.append('../..')
from model_tests.util import *

from .MNISTNet import MNISTNet
from .CIFAR10Net import CIFAR10Net, CIFAR10_Noise_Net
from .CIFAR100Net import CIFAR100Net