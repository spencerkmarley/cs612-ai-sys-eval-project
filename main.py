# Import all standard libraries used in course 

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

# Import model tests
import model_tests.check_file as cf
import model_tests.robustness as rb

# Import models
import models.train_model
from models.train_model import MNISTNet
mnist_model = MNISTNet()
mnist_model.load_state_dict(torch.load('models/mnist.pt'))

# Import datasets
mnist_dataset = datasets.MNIST('data/', train=False, transform=transforms.ToTensor())

# Check that it is a PyTorch file
cf.check_file("models/mnist.pt")

# Test robustness of model
rb.test_robust(mnist_model, mnist_dataset, 1)
