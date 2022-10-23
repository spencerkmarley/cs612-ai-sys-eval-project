import numpy as np
import torch

from torch import Tensor
from typing import Any

def l1_norm(x: Tensor, y: Tensor) -> Tensor:
    """ Compute the L1 norm between two tensors """
    res = torch.abs(x - y)
    return torch.sum(res)

def l2_norm(x, y):
    """ Compute the L2 norm between two tensors """
    res = torch.sum((x - y) ** 2)
    return torch.sqrt(res)

def linf_norm(x, y):
    """ Compute the L-inf norm between two tensors """
    res = torch.max(torch.abs(x - y))
    return res

def get_pytorch_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

def add_noise(weights, noise, device = None):
    """ Add the noise vector to the weights """
    if device is None:
        device = get_pytorch_device()
        
    with torch.no_grad():
        weights.add_(noise.to(device))

def main():
    torch.manual_seed(42)
    x = torch.randn(2,2)
    y = torch.randn(2,2)
    a = linf_norm(x,y)
    
    print(f'Tensor x:\n{x}\n')
    print(f'Tensor y:\n{y}\n')
    print(f'L-inf norm: {a}')
    pass

if __name__ == '__main__':
    main()
