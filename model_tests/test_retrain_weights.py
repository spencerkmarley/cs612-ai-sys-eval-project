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

from torchsummary import summary

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

import os
import pathlib

# import util

"""Clean model"""

class CIFAR10Net(nn.Module):
    # from https://www.kaggle.com/code/shadabhussain/cifar-10-cnn-using-pytorch
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2) # output: 64 x 16 x 16

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2) # output: 128 x 8 x 8

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2) # output: 256 x 4 x 4

        self.fc1 = nn.Linear(256 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = F.relu(x)
        x = self.pool3(x)

        x = torch.flatten(x, 1)        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        output = x
        return output
    
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

def train(model, dataloader, loss_fn, optimizer, device):
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
            print('loss: {:.4f} [{}/{}]'.format(loss, current, size))
            
def test(model, dataloader, loss_fn, device):
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
    print('Test Result: Accuracy @ {:.2f}%, Avg loss @ {:.4f}\n'.format(accuracy, loss))

    return accuracy, loss


def main():
    """Load subject model and get subject model's summary and weights"""

    # Device selection - includes Apple Silicon
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    # Load the model to be tested for the presence of a potential backdoor
    subject_model = load_model(CIFAR10Net, './models/best_model_CIFAR10_10BD.pt', device)
    subject_model_weights = get_weights_from_model(subject_model)
    subject_fc3_weights = subject_model_weights['fc3.weight'][0]
    subject_fc3_bias = subject_model_weights['fc3.bias'][0]

    """Retrain subject model"""
    transform = transforms.ToTensor()

    train_kwargs = {'batch_size': 100, 'shuffle':True}
    test_kwargs = {'batch_size': 1000}
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(testset, **test_kwargs)

    """Testing subject model against clean test data"""

    subject_test_accuracy, subject_test_loss = test(subject_model,test_loader,nn.CrossEntropyLoss(),device)

    """Retraining subject model for testing"""
    
    n_control_models = 2 # Number of control models to build to check for deviation
    
    retrain_models = []
    for n in range(n_control_models):
        FORCE_RETRAIN = True # Only set to True if you want to retrain the model
        if not os.path.exists('./models/retrained_CIFAR10_10BD_'+str(n)+'.pt') or FORCE_RETRAIN:
            print(f'Training #{n+1} of {n_control_models} models')
            retrain_model = CIFAR10Net().to(device)
            optimizer = optim.Adam(retrain_model.parameters(), lr=0.001)
            epochs = 30
            best_accuracy = 0

            for epoch in range(epochs):
                print('\n------------- Epoch {} -------------\n'.format(epoch+1))
                train(retrain_model, train_loader, nn.CrossEntropyLoss(), optimizer, device)
                accuracy, loss = test(retrain_model, test_loader, nn.CrossEntropyLoss(), device)

                #Callback to save model with lowest loss
                if accuracy > best_accuracy:
                    print(f'Saving model with new best accuracy: {accuracy:.4f}')
                    save_model(retrain_model,'./models/retrained_CIFAR10_10BD_'+str(n)+'.pt')
                    best_accuracy = accuracy
        
        # Regardless of whether we retrained the model, load it so we have the best model saved
        retrain_model = load_model(CIFAR10Net, './models/retrained_CIFAR10_10BD_'+str(n)+'.pt',device=device)
        retrain_models.append(retrain_model)

    """Compare weights and biases of the feature vector layer."""
    retrain_model = retrain_models[0]
    test(retrain_model,test_loader,nn.CrossEntropyLoss(),device)
    
    # TEST DIFFERENCES BETWEEN RETRAINED MODELS
    subject_model = retrain_models[1]
    subject_model_weights = get_weights_from_model(subject_model)
    subject_fc3_weights = subject_model_weights['fc3.weight'][0]
    subject_fc3_bias = subject_model_weights['fc3.bias'][0]

    """The accuracy and loss of the retrained model is better than the subject model. To investigate the weights of the final linear layer further."""
    retrain_params = retrain_model.state_dict()
    retrain_fc3_weights = retrain_params['fc3.weight'][0]
    retrain_fc3_bias = retrain_params['fc3.bias'][0]

    Weight_delta = retrain_fc3_weights-subject_fc3_weights
    Bias_delta = retrain_fc3_bias-subject_fc3_bias

    q75, q25 = np.percentile(Weight_delta.to('cpu').numpy(), [75 ,25])
    iqr = q75 - q25
    maxbound = q75+1.5*iqr
    minbound = q25-1.5*iqr

    plt.figure(figsize=(20,5))
    plt.plot(np.arange(1,513),Weight_delta.to('cpu').numpy(),'^--k')
    plt.axhline(maxbound,0,512)
    plt.axhline(minbound,0,512)
    plt.ylabel('Retrain-subject weight delta at last linear layer')
    plt.xlabel('Neuron number')
    plt.show()

    num_outlier_neurons = sum(Weight_delta>maxbound)+sum(Weight_delta<minbound)
    percent_outlier_neurons = num_outlier_neurons/len(Weight_delta)
    print(f'Number of outlier neurons: {num_outlier_neurons.item()}')
    print(f'Percentage of outlier neurons: {percent_outlier_neurons.item()*100:.2f}%')

    threshold = 0.10 #can be adjusted.

    if percent_outlier_neurons > threshold:
        print(f'It is possible that the network has a backdoor, becuase the percentage of outlier neurons is above the {threshold} threshold.')
    else:
        print('It is unlikely that the network has a backdoor.')
        
if __name__ == '__main__':
    main()
