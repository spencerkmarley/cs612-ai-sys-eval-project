import torch

from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from torchsummary import summary

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

import os
import pathlib
import sys

from collections import defaultdict, Counter

from copy import deepcopy

#torch.manual_seed(42)

# Add paths
sys.path.append('.')

# Utility functions
import util
from util import get_pytorch_device, open_model, load_model, save_model, train, test

# Import models with perturbations
from models.definitions import CIFAR10_Noise_Net, CIFAR10Net_NeuronsOff


#
# Function definitions
#
def has_backdoor(subject_model, test_model, threshold=0.1):
  '''
  testing_model: a .pt model, the finetuned subject_model
  threshold: percentage [0,1] threshold to flag whether a class could have a backdoor
  test_type: a string, 'Gaussian noised', 'randomly switching off neurons', 'neural attention distilled'
  '''
  percent_diff = prediction_variance(subject_model, test_model)
  print(f'Percentage difference in inferences: {percent_diff}')

  percent_diff = dict(sorted(percent_diff.items(), key=lambda item: abs(item[1]),reverse=True))

  backdoored_classes = [k for k,v in percent_diff.items() if abs(v)>=threshold]
  
  if len(backdoored_classes):
    print('\nThe subject model most likely has a backdoor')
    print('\n----Most likely backdoored classes by descending order----')
    print(' '.join(str(k) for k in backdoored_classes))
  else:
    print('\nThe subject model most likely does not have a backdoor')
  
  return backdoored_classes


def prediction_variance(subject_model, test_model):
  '''
  Computes the prediction difference between the subject model and the test model.
  Returns a dictionary of the absolute difference in the range of [0,1]
  '''
  
  dist_subject = util.model.get_pred_distribution(subject_model, test_loader, device)
  dist_test = util.model.get_pred_distribution(test_model, test_loader, device)
  
  pred_diff = {x: abs(dist_test[x] - dist_subject[x])/dist_subject[x] if dist_subject[x] else 0 for x in dist_test}
  return pred_diff


# Load the subject model
from models.definitions import CIFAR10Net
subject_model_filename = 'models/subject/best_model_CIFAR10_10BD.pt'
subject_model = load_model(CIFAR10Net, subject_model_filename)

#subject_model = open_model(subject_model_filename)
print(summary(subject_model,(3,32,32)))



# Train and test arguments
transform = transforms.ToTensor()

train_kwargs = {'batch_size': 100, 'shuffle':True}
test_kwargs = {'batch_size': 1000}
trainset = datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
testset = datasets.CIFAR10(root='data', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, **train_kwargs)
test_loader = torch.utils.data.DataLoader(testset, **test_kwargs)

device = get_pytorch_device()

# Test the accuracy of the subject model loaded
loss_fn = nn.CrossEntropyLoss()
test(subject_model, test_loader, loss_fn, device)



pred_distribution = util.model.get_pred_distribution(subject_model, test_loader, device)
print(f' Class distribution from inference of clean model: {pred_distribution}')


def retrain_model(
  base_model_filename,  # Location of the base PyTorch model to be loaded from file
  retrain_arch,  # Model architecture to retrain - found in models/definitions
  save_filename,  # Location to save the retrained model
  train_loader,  # Training data loader
  device = None,  # Device to use for training (if None, use get_pytorch_device())
  epochs = 30,  # Training epochs
  lr = 0.001,  # Learning rate
  force_retrain = False,  # Force retrain even if model file exists
  ):
  """ Retrains a model from the base with alterations from the retrain_arch """
  
  if device is None:
    device = get_pytorch_device()
  
  if force_retrain or not os.path.exists(save_filename):
    new_model = load_model(retrain_arch, base_model_filename)
    optimizer = optim.Adam(new_model.parameters(), lr=lr)
    best_accuracy = 0
    
    for epoch in range(epochs):
      print('\n------------- Epoch {} -------------\n'.format(epoch+1))
      train(new_model, train_loader, nn.CrossEntropyLoss(), optimizer, device)
      accuracy, _ = test(new_model, test_loader, nn.CrossEntropyLoss(), device)

      if accuracy > best_accuracy:
        save_model(new_model, save_filename)
        best_accuracy = accuracy
        
  new_model = load_model(retrain_arch, save_filename)
  return new_model

  
#
# TEST 1 - Gaussian noise
#
save_filename = os.path.join('models', 'CIFAR10NET_noised.pt')
model_noise = retrain_model(
  base_model_filename = subject_model_filename,
  retrain_arch = CIFAR10_Noise_Net,
  save_filename = save_filename,
  train_loader = train_loader,
  force_retrain = False,
)

# if FORCE_RELOAD or not os.path.exists(save_filename):
#     model_noise = load_model(CIFAR10_Noise_Net, subject_model_filename)
#     optimizer = optim.Adam(model_noise.parameters(),lr = 0.001)
#     epochs = 30
#     best_accuracy = 0

#     for epoch in range(epochs):
#         print('\n------------- Epoch {} -------------\n'.format(epoch+1))
#         train(model_noise, train_loader, nn.CrossEntropyLoss(), optimizer, device)
#         accuracy, loss = test(model_noise, test_loader, nn.CrossEntropyLoss(), device)

#         if accuracy > best_accuracy:
#             save_model(model_noise, save_filename)
#             best_accuracy = accuracy
    
# model_noise = load_model(CIFAR10_Noise_Net, save_filename)

backdoored_classes = has_backdoor(subject_model, model_noise)
if len(backdoored_classes):
    print('The subject model has a backdoor')
    print(backdoored_classes)

#
# TEST 2 - Randomly switch off neurons
#
FORCE_RELOAD = False
save_filename = os.path.join('models', 'CIFAR10NET_NeuronsOff.pt')

model_NeuronsOff = load_model(CIFAR10Net_NeuronsOff, subject_model_filename)
optimizer = optim.Adam(model_NeuronsOff.parameters(),lr = 0.001)
epochs =30
best_accuracy = 0

for epoch in range(epochs):
  print('\n------------- Epoch {} -------------\n'.format(epoch+1))
  train(model_NeuronsOff, train_loader, nn.CrossEntropyLoss(), optimizer, device)
  accuracy, loss, output = test(model_NeuronsOff, test_loader, nn.CrossEntropyLoss(), device)

  # Callback to save model with lowest loss
  if accuracy > best_accuracy:
    save_model(model_NeuronsOff, save_filename)
    best_accuracy = accuracy
    
model_NeuronsOff = load_model(CIFAR10Net_NeuronsOff, save_filename)
backdoored_classes = has_backdoor(subject_model, model_noise)
if len(backdoored_classes):
    print('The subject model has a backdoor')
    print(backdoored_classes)