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
from model_tests import test_retrain_weights as rt
from models.train import train_mnist

# Load model definitions
import models
from models.definitions import MNISTNet, CIFAR10Net, CIFAR100Net

# Provide filepaths
data_file_path = "data/"

# Provide test cases
TEST_CASE = 1

if TEST_CASE == 1:
    network_definition = MNISTNet()
    benign_model_file_path = "models/benign/mnist.pt"
    subject_model_file_path = "models/subject/mnist_backdoored_1.pt"
    dataset = datasets.MNIST(data_file_path, train=False, transform=transforms.ToTensor())
    
elif TEST_CASE == 2:
    network_definition = CIFAR10Net()
    benign_model_file_path = "models/benign/benign_CIFAR10.pt"
    subject_model_file_path = "./models/subject/best_model_CIFAR10_10BD.pt"

# Set parameters
NUM_IMG = 10
EPS = 0.2
THRESHOLD = 0.3

# Import benign model
benign_model = network_definition
benign_model.load_state_dict(torch.load(benign_model_file_path))

# Import subject model
subject_model = network_definition
subject_model.load_state_dict(torch.load(subject_model_file_path))

# Retrain the subject model and test the weights to deteremine if there is a back door
# rt.main(network=network_definition, subject=subject_model, trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor()), testset = datasets.CIFAR10(root='./data', train=False,download=True, transform=transforms.ToTensor()), retrained="./models/retrained/retrained_CIFAR10_10BD_", n_control_models = 2, verbose=True)

# Test robustness of model using robustness.py tests to determine if there is a backdoor
for i in range(13):
    try:
        robust = rb.test_robust(benign=benign_model, subject=subject_model, dataset=dataset, test=i, num_img=NUM_IMG, eps=EPS, threshold=THRESHOLD, verbose=False)
        print(robust)
    except:
        print("Robustness test " + str(i) + " failed")
    
# Fine tuning tests - gaussian noise, retraining with dropout, neural attention distillation (which classes have backdoor)
# backdoor_forgetting.ipynb

# Regenerate the trigger
# trigger_synthesis standard.py