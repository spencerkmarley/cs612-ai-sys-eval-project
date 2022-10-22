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
from model_tests import robustness as rb

# Import baseline models
from models.train import train_mnist
from models.train.train_mnist import MNISTNet 

mnist_model = MNISTNet()
mnist_model.load_state_dict(torch.load('models/mnist.pt'))

# Import subject models
subject_model = MNISTNet()
subject_model.load_state_dict(torch.load('models/mnist.pt'))

# Import datasets
mnist_dataset = datasets.MNIST('data/', train=False, transform=transforms.ToTensor())

# Test robustness of model
# for i in range(12):
#     rb.test_robust(benign=mnist_model, subject=subject_model, dataset=mnist_dataset, test=i, num_img=10)

NUM_IMG = 10
EPS = 0.2
THRESHOLD = 0.3

if rb.test_robust(benign=mnist_model, subject=subject_model, dataset=mnist_dataset, test=1, num_img=NUM_IMG, eps=EPS, threshold=THRESHOLD, verbose=True):
    print("Model is robust")
else:
    print("Model is not robust")