"""Setup libraries"""

import torch

from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# Utility functions
from util import config as c
from util import get_pytorch_device
from util import logger

#from torchsummary import summary

import matplotlib.pyplot as plt
#import seaborn as sns

import numpy as np

import os
import pathlib
from models.definitions import MNISTNet, CIFAR10Net, CIFAR100Net

VERBOSE = c.VERBOSE

# FUNCTION DEFINITIONS
def load_model(model_class, name, device):
    model = model_class()
    model.load_state_dict(torch.load(name, map_location=device))
    model.to(device)

    return model

def save_model(model, name):
    p = pathlib.Path(name)
    if not os.path.exists(p.parent):
        os.makedirs(p.parent, exist_ok=True)
    torch.save(model.state_dict(), name)

def get_weights_from_model(model):
    """ Given a PyTorch model, return a dictionary of the weights and biases """
    weights = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            weights[name] = param.data
    return weights

def train(model, dataloader, loss_fn, optimizer, device, verbose=VERBOSE):
    size = len(dataloader.dataset)
    model.train()

    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        # Compute prediction error
        pred = model(x)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(x)
            if verbose:
                logger.info('loss: {:.4f} [{}/{}]'.format(loss, current, size))
            
def test(model, dataloader, loss_fn, device, verbose=False):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.to(device)
    model.eval()
    loss, correct = 0.0, 0
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            pred = model(x)
            loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.int).sum().item()
    
    loss /= num_batches
    correct /= size
    accuracy = 100*correct
    if verbose:
        logger.info('Test Result: Accuracy @ {:.2f}%, Avg loss @ {:.4f}'.format(accuracy, loss))

    return accuracy, loss

def main(network, 
         subject, 
         trainset, 
         testset, 
         retrained, 
         n_control_models, 
         model_string,
         learning_rate, 
         epochs, 
         threshold, 
         verbose):
    """Load subject model and get subject model's summary and weights"""
    
    logger.info('Retraining the subject model and test the weights to determine if there is a backdoor')
    device = get_pytorch_device()
    logger.debug(f'Using device: {device}')

    # Load the model to be tested for the presence of a potential backdoor
    # subject_model = load_model(network, subject, device)
    subject_model = subject

    """Retrain subject model"""
    train_kwargs = {'batch_size': 100, 'shuffle':True}
    test_kwargs = {'batch_size': 1000}
    train_loader = torch.utils.data.DataLoader(trainset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(testset, **test_kwargs)

    """Testing subject model against clean test data"""

    subject_test_accuracy, _ = test(subject_model,test_loader,nn.CrossEntropyLoss(),device,verbose)
    if verbose:
        logger.info('Subject model test accuracy {:.2f}%'.format(subject_test_accuracy))

    """Retraining subject model for testing"""
    
    retrain_models = []
    for n in range(n_control_models): # Number of control models to build to check for deviation
        FORCE_RETRAIN = c.FORCE_RETRAIN # Only set to True if you want to retrain the model
        path = retrained+str(n)+'.pt'
        if verbose:
            logger.info(f'Model path: {path}')
        if not os.path.exists(path) or FORCE_RETRAIN:
            if verbose:
                logger.info(f'Training #{n+1} of {n_control_models} models')
            retrain_model = network.to(device)
            optimizer = optim.Adam(retrain_model.parameters(), lr=learning_rate)
            best_accuracy = 0

            for epoch in range(epochs):
                if verbose:
                    logger.info('\n------------- Epoch {} -------------\n'.format(epoch+1))
                train(retrain_model, train_loader, nn.CrossEntropyLoss(), optimizer, device, verbose)
                accuracy, loss = test(retrain_model, test_loader, nn.CrossEntropyLoss(), device, verbose)

                #Callback to save model with lowest loss
                if accuracy > best_accuracy:
                    if verbose:
                        logger.info(f'Saving model with new best accuracy: {accuracy:.2f}%')
                    save_model(retrain_model, path)
                    best_accuracy = accuracy
        
        # Regardless of whether we retrained the model, load it so we have the best model saved
        
        if model_string == 'MNIST':
            retrain_model = load_model(MNISTNet, path, device=device)
        elif model_string =='CIFAR10':
            retrain_model = load_model(CIFAR10Net, path, device=device)
        else:
            retrain_model = load_model(CIFAR100Net, path,device=device)
        retrain_models.append(retrain_model)

    """Compare weights and biases of the feature vector layer."""
    retrain_model = retrain_models[0]
    test(retrain_model,test_loader,nn.CrossEntropyLoss(),device, verbose)
    
    # TEST DIFFERENCES BETWEEN RETRAINED MODELS
    retrain_models[1] = subject_model
    subject_model_weights = get_weights_from_model(subject_model)
    
    if model_string == 'MNIST':
        subject_weights = subject_model_weights['fc4.weight']
        subject_bias = subject_model_weights['fc4.bias']
        retrain_params = retrain_model.state_dict()
        retrain_weights = retrain_params['fc4.weight']
        retrain_bias = retrain_params['fc4.bias']
    
    elif model_string == 'CIFAR10':
        subject_weights = subject_model_weights['fc3.weight'][0]
        subject_bias = subject_model_weights['fc3.bias'][0]
        retrain_params = retrain_model.state_dict()
        retrain_weights = retrain_params['fc3.weight'][0]
        retrain_bias = retrain_params['fc3.bias'][0]
    
    else:
        subject_weights = subject_model_weights['res2.1.1.weight']
        subject_bias = subject_model_weights['res2.1.1.bias']
        retrain_params = retrain_model.state_dict()
        retrain_weights = retrain_params['res2.1.1.weight']
        retrain_bias = retrain_params['res2.1.1.bias']

    Weight_delta = retrain_weights-subject_weights
    Bias_delta = retrain_bias-subject_bias

    q75, q25 = np.percentile(Weight_delta.to('cpu').numpy(), [75 ,25])
    iqr = q75 - q25
    maxbound = q75+1.5*iqr
    minbound = q25-1.5*iqr

    # plt.figure(figsize=(20,5))
    # plt.plot(np.arange(1,513),Weight_delta.to('cpu').numpy(),'^--k')
    # plt.axhline(maxbound,0,512)
    # plt.axhline(minbound,0,512)
    # plt.ylabel('Retrain-subject weight delta at last linear layer')
    # plt.xlabel('Neuron number')
    # plt.show()
    
   
    num_outlier_neurons = sum(Weight_delta>maxbound)+sum(Weight_delta<minbound)
    percent_outlier_neurons = num_outlier_neurons/len(Weight_delta)

    if verbose:
        if num_outlier_neurons.numel()>1:
            # for i, num in enumerate(list(num_outlier_neurons.numpy())):
            for i, num in enumerate(num_outlier_neurons.tolist()):
                if verbose:
                    logger.info(f'Number of outlier neurons for class {i}: {num}')
            # for i, num in enumerate(list(percent_outlier_neurons.numpy())):
            for i, num in enumerate(percent_outlier_neurons.tolist()):
                if verbose:
                    logger.info(f'Percentage of outlier neurons for class {i}: {num*100:.2f}%')
        elif num_outlier_neurons.numel()==1:
            if verbose:
                logger.info(f'Number of outlier neurons: {num_outlier_neurons.cpu().numpy()}')
                logger.info(f'Percentage of outlier neurons: {percent_outlier_neurons.cpu().numpy()*100:.2f}%')
        else:
            if verbose:
                logger.info('No outlier neurons')

    if percent_outlier_neurons.numel()>1:
        if any (percent_outlier_neurons) > threshold:
            backdoor = True
            if verbose:
                logger.info(f'It is possible that the network has a backdoor, because the percentage of outlier neurons is above the {threshold} threshold.')
        else:
            backdoor = False
            if verbose:
                logger.info('It is unlikely that the network has a backdoor.')
                
    elif percent_outlier_neurons.numel()==1:
        if percent_outlier_neurons.cpu().numpy()>threshold:
            backdoor = True
            if verbose:
                logger.info(f'It is possible that the network has a backdoor because the percentage of outlier neurons is above the {threshold} threshold.')
        else:
            backdoor = False
        
    else:
        return False

    return backdoor

if __name__ == '__main__':
    main()
