import pathlib
import torch
from collections import Counter

from .pytorch_functions import get_pytorch_device

def add_noise(weights, noise, device = None):
    """ Add the noise vector to the weights """
    if device is None:
        device = get_pytorch_device()


    with torch.no_grad():
        weights.add_(noise.to(device))


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


