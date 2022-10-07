import torch

from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import matplotlib.pyplot as plt


def test_robust(model, dataset, test):
    """
    Test the robustness of a model by:
    (i) taking clean samples
    (ii) perturbing them by a perturbation that we know doesn't change the label
    (iii) using the benign and subject models to classify the pertubred images
    (iv) concluding that there is a backdoor if we discover discrepancies
    """
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    
    robust = True

    if test == 0:
        perturb_rotation(dataset)
    elif test == 1:
        perturb_change_pixels(dataset)
    elif test == 2:
        perturb_invert(dataset)
    elif test == 3:
        perturb_change_lighting(dataset)
    elif test == 4:
        perturb_zoom_in_out(dataset)
    elif test == 5:
        perturb_resize(dataset)
    elif test == 6:
        perturb_crop_rescale(dataset)
    elif test == 7:
        perturb_bit_depth_reduction(dataset)
    elif test == 8:
        perturb_compress_decompress(dataset)
    elif test == 9:
        perturb_total_var_min(dataset)
    elif test == 10:
        perturb_adding_noise(dataset)
    elif test == 11:
        perturb_watermark(dataset)
    else:
        print("Please provide a valid test number")
    
    return robust


def perturb_rotation(dataset):
    # Perturb some clean samples by rotating them
    print("Perturbing by rotation...")
    return dataset

def perturb_change_pixels(dataset):
    # Perturb some clean samples by changing pixels
    print("Perturbing by changing pixels...")
    return dataset

def perturb_invert(dataset):
    # Perturb some clean samples by inverting them
    print("Perturbing by inverting images...")
    return dataset

def perturb_change_lighting(dataset):
    # Perturb some clean samples by changing the lighting
    print("Perturbing by changing the lighting...")
    return dataset

def perturb_zoom_in_out(dataset):
    # Perturb some clean samples by zooming in and out
    print("Perturbing by zooming in and out...")
    return dataset

def perturb_resize(dataset):
    # Perturb some clean samples by resizing
    print("Perturbing by resizing...")
    return dataset

def perturb_crop_rescale(dataset):
    # Perturb some clean samples by cropping and rescaling
    print("Perturbing by cropping and rescaling...")
    return dataset

def perturb_bit_depth_reduction(dataset):
    # Perturb some clean samples by bit depth reduction
    print("Perturbing by bit depth reduction...")
    return dataset

def perturb_compress_decompress(dataset):
    # Perturb some clean samples by compressing and decompressing
    print("Perturbing by compressing and decompressing...")
    return dataset

def perturb_total_var_min(dataset):
    # Perturb some clean samples by total var min
    print("Perturbing by total var min...")
    return dataset

def perturb_adding_noise(dataset):
    # Perturb some clean samples by adding noise
    print("Perturbing by adding noise...")
    return dataset

def perturb_watermark(dataset):
    # Perturb some clean samples by adding a watermark
    print("Perturbing by adding a watermark...")
    return dataset

