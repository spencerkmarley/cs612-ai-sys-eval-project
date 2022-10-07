import torch

from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

# Custom libraries
import model_tests.check_file as cf

# Check that it is a PyTorch model
cf.check_file("model/CIFAR10_10BD.pt")

