import os
import pathlib
import sys
import torch
from collections import Counter

sys.path.append('.')
from .pytorch_functions import get_pytorch_device

sys.path.append('..')
from models.definitions import *

        
def open_model(model_filename):
    """ Opens a model from the file, checks that it is a pytorch model, and assigns it to the correct architecture.
    Currently only supporting MNIST, CIFAR10, CIFAR1000 per project requirements """
    
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


def load_model(model_class, name):
    model = model_class()
    model.load_state_dict(torch.load(name, map_location=get_pytorch_device()))

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
            


def test(model, dataloader, loss_fn, device, testset = None):
    """ Run test on the model"""
    # Specific for some tests requiring the testset to be specified
    if testset is not None:
        output = {k:0 for k in list(dict(Counter(testset.targets)).keys())}

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
            if testset is not None:
                for res in result:
                    output[res.to('cpu').numpy().item()]+=1

    loss /= num_batches
    correct /= size
    accuracy = 100*correct
    print('Test Result: Accuracy @ {:.2f}%, Avg loss @ {:.4f}\n'.format(accuracy, loss))

    if testset is not None:
        return accuracy, loss, output
    else:
        return accuracy, loss


def main():
    model_filename = 'models/subject/mnist_backdoored_1.pt'
    model = open_model(model_filename)
    pass

if __name__ == '__main__':
    main()