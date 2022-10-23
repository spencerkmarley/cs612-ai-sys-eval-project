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

# Import custom libraries
from model_tests import robustness as rb
from models.train import train_mnist

# Load model definitions
import models
from models.definitions import MNISTNet, CIFAR10Net, CIFAR100Net

# Provide filepaths
benign_model_file_path = "models/benign/mnist.pt"
subject_model_file_path = "models/subject/mnist_backdoored_1.pt"
data_file_path = "data/"

# Set parameters
NUM_IMG = 10
EPS = 0.2
THRESHOLD = 0.3

# Import benign model(s)
benign_model = MNISTNet()
benign_model.load_state_dict(torch.load(benign_model_file_path))

# Import subject model(s)
subject_model = MNISTNet()
subject_model.load_state_dict(torch.load(subject_model_file_path))

# Import dataset(s)
mnist_dataset = datasets.MNIST(data_file_path, train=False, transform=transforms.ToTensor())

# Retrain the subject model and test the weights to deteremine if there is a back door
## TO DO
# test_retrain_weights.py

# Test robustness of model using robustness.py tests to determine if there is a backdoor
for i in range(12):
    rb.test_robust(benign=benign_model, subject=subject_model, dataset=mnist_dataset, test=i, num_img=NUM_IMG, eps=EPS, threshold=THRESHOLD, verbose=True)

# Fine tuning tests - gaussian noise, retraining with dropout, neural attention distillation (which classes have backdoor)
## TO DO 
# Gerry trying to port backdoor_forgetting.ipynb

# Regenerate the trigger
## TO DO
# trigger_synthesis standard.py