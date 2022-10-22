import torch

def load_model(model_class, name):
    model = model_class()
    model.load_state_dict(torch.load(name, map_location=get_pytorch_device()))

    return model