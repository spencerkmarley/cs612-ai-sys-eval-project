import os
import pathlib
import sys
import torch
from collections import Counter, defaultdict

sys.path.append('.')
from .pytorch_functions import get_pytorch_device

sys.path.append('..')
from models.definitions import *

        
def open_model(model_filename, device=None):
    """ Opens a model from the file, checks that it is a pytorch model, and assigns it to the correct architecture.
    Currently only supporting MNIST, CIFAR10, CIFAR1000 per project requirements """
    
    if device is None:
        device = get_pytorch_device()
    
    # Check that file exists
    if not os.path.exists(model_filename):
        raise FileNotFoundError(f'File {model_filename} does not exist')
    
    # Try opening it using torch.load to check it is a pytorch file
    model_file = torch.load(model_filename, map_location=get_pytorch_device())
    
    # Check which class it belongs
    for model_arch in [MNISTNet, CIFAR10Net, CIFAR100Net, MNISTNet]:
        try:
            model = model_arch()
            model.load_state_dict(model_file)
            model.to(get_pytorch_device())
            return model
        except:
            pass
        
    raise ValueError(f'File {model_filename} does not belong to an implemented architecture')
    pass


def load_model(model_class, name, device=None):
    if device is None:
        device = get_pytorch_device()
        
    model = model_class()
    model.load_state_dict(torch.load(name, map_location=get_pytorch_device()))
    model.to(get_pytorch_device())
    return model


def save_model(model, name):
    p = pathlib.Path(name)
    if not os.path.exists(p.parent):
        os.makedirs(p.parent, exist_ok=True)

    torch.save(model.state_dict(), name)


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
    """ Run test on the model"""
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
            result = pred.argmax(1)
            correct += (result == y).type(torch.int).sum().item()

    loss /= num_batches
    correct /= size
    accuracy = 100*correct
    print('Test Result: Accuracy @ {:.2f}%, Avg loss @ {:.4f}\n'.format(accuracy, loss))

    return accuracy, loss

    
def get_pred_distribution(model, dataloader, device):
    """ Given a model and dataloader object, return a dictionary of the distribution """
    res_distribution = defaultdict(int)
    
    # Get the indices of the target classes
    for i in dataloader.dataset.class_to_idx.values():
        res_distribution[i] = 0
    
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.to(device)
    model.eval()
    
    preds = []
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            preds.extend(pred.argmax(1).tolist())
            
    for k,v in dict(Counter(preds)).items():
        res_distribution[k] += v
        
    return dict(res_distribution)
    pass


def main():
    model_filename = 'models/subject/mnist_backdoored_1.pt'
    model = open_model(model_filename)
    pass

if __name__ == '__main__':
    main()