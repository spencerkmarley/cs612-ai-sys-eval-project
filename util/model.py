import torch

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

