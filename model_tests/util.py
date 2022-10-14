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


def main():
    torch.manual_seed(42)
    x = torch.randn(2,2)
    y = torch.randn(2,2)
    a = l2_norm(x,y)
    
    print(f'Tensor x:\n{x}\n')
    print(f'Tensor y:\n{y}\n')
    print(f'L2 norm: {a}')
    pass

if __name__ == '__main__':
    main()