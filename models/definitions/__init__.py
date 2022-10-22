# Import utility functions from model_tests folder
import sys
sys.path.append('../..')
from model_tests.util import *

# Standard architectures
from .MNISTNet import MNISTNet
from .CIFAR10Net import CIFAR10Net
from .CIFAR100Net import CIFAR100Net

# Perturbed architectures
from .MNISTNet import MNIST_Noise_Net, MNISTNet_NeuronsOff, MNISTNet_AT
from .CIFAR10Net import CIFAR10_Noise_Net, CIFAR10Net_NeuronsOff, CIFAR10Net_AT
from .CIFAR100Net import CIFAR100_Noise_Net, CIFAR100Net_NeuronsOff, CIFAR100Net_AT