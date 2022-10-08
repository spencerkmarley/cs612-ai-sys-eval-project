# Import all standard libraries used in course 

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

"""
Test the robustness of a model by:
(i) taking clean samples
(ii) perturbing them by a perturbation that we know doesn't change the label
(iii) using the subject model to classify the pertubred images
(iv) concluding that there is a backdoor if we discover discrepancies
"""

def test_robust(benign, subject, dataset, test, num_img, eps, threshold):

    robust = True

    if test == 0:
        perturb_rotation(benign, subject, dataset, num_img)
    elif test == 1:
        perturb_change_pixels(benign, subject, dataset, num_img, eps, threshold)
    elif test == 2:
        perturb_invert(benign, subject, dataset, num_img)
    elif test == 3:
        perturb_change_lighting(benign, subject, dataset, num_img)
    elif test == 4:
        perturb_zoom_in_out(benign, subject, dataset, num_img)
    elif test == 5:
        perturb_resize(benign, subject, dataset, num_img)
    elif test == 6:
        perturb_crop_rescale(benign, subject, dataset, num_img)
    elif test == 7:
        perturb_bit_depth_reduction(benign, subject, dataset, num_img)
    elif test == 8:
        perturb_compress_decompress(benign, subject, dataset, num_img)
    elif test == 9:
        perturb_total_var_min(benign, subject, dataset, num_img)
    elif test == 10:
        perturb_adding_noise(benign, subject, dataset, num_img)
    elif test == 11:
        perturb_watermark(benign, subject, dataset, num_img)
    else:
        print("Please provide a valid test number")
    
    return robust


def perturb_rotation(benign, subject, dataset, num_img):
    # Perturb some clean samples by rotating them
    print("Perturbing by rotation...")
    return dataset

def perturb_change_pixels(benign, subject, dataset, num_img, eps, threshold):
    """
    Perturb some clean samples by changing pixels
    """

    robust = True

    # Take clean samples
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    count = 0
    discrepancies = 0
    samples = 0

    # Perturb them by a perturbation that we know doesn't change the label
    for x, y in test_loader:
        same_label = False

        if count < num_img:
            
            # Perturb the clean sample
            x_perturb = x.detach().clone()
            x_perturb.requires_grad = True
            prediction = benign(x_perturb)
            loss = F.cross_entropy(prediction, y)
            loss.backward()
            grad_data = x_perturb.grad.data
            x_perturb = torch.clamp(x_perturb + eps * grad_data.sign(), 0, 1).detach()

            # Check if it changes the label
            prediction_benign = benign(x_perturb)
            if prediction.argmax(1) == prediction_benign.argmax(1):
                same_label = True
                samples += 1

            # Use subject model to classify the pertubred images
            if same_label:
                prediction_subject = subject(x_perturb)
                
                if prediction_benign.argmax(1) != prediction_subject.argmax(1):
                    discrepancies += 1

            count += 1

    # Conclude that there is a backdoor if we discover discrepancies    
    if discrepancies/samples >= threshold:
        robust = False

    return robust

def perturb_invert(benign, subject, dataset, num_img):
    # Perturb some clean samples by inverting them
    print("Perturbing by inverting images...")
    return dataset

def perturb_change_lighting(benign, subject, dataset, num_img):
    # Perturb some clean samples by changing the lighting
    print("Perturbing by changing the lighting...")
    return dataset

def perturb_zoom_in_out(benign, subject, dataset, num_img):
    # Perturb some clean samples by zooming in and out
    print("Perturbing by zooming in and out...")
    return dataset

def perturb_resize(benign, subject, dataset, num_img):
    # Perturb some clean samples by resizing
    print("Perturbing by resizing...")
    return dataset

def perturb_crop_rescale(benign, subject, dataset, num_img):
    # Perturb some clean samples by cropping and rescaling
    print("Perturbing by cropping and rescaling...")
    return dataset

def perturb_bit_depth_reduction(benign, subject, dataset, num_img):
    # Perturb some clean samples by bit depth reduction
    print("Perturbing by bit depth reduction...")
    return dataset

def perturb_compress_decompress(benign, subject, dataset, num_img):
    # Perturb some clean samples by compressing and decompressing
    print("Perturbing by compressing and decompressing...")
    return dataset

def perturb_total_var_min(benign, subject, dataset, num_img):
    # Perturb some clean samples by total var min
    print("Perturbing by total var min...")
    return dataset

def perturb_adding_noise(benign, subject, dataset, num_img):
    # Perturb some clean samples by adding noise
    print("Perturbing by adding noise...")
    return dataset

def perturb_watermark(benign, subject, dataset, num_img):
    # Perturb some clean samples by adding a watermark
    print("Perturbing by adding a watermark...")
    return dataset


#     if y_adv != y:
#         x = x.detach().numpy().reshape(-1)
#         x_adv = x_adv.detach().numpy().reshape(-1)

#         y, y_adv = y.item(), y_adv.item()

#         print('\nFound an adversarial sample!!!\n')

#         print('pred adv = {}'.format(pred_adv.detach().numpy().reshape(-1)))
#         print('lbl adv = {}\n'.format(y_adv))

#         display(x, y, x_adv, y_adv)
#         return True
#     else:
#         print('\nCan\'t find adversarial samples!!!\n')
#         return False

# num_img, num_adv, eps = 0, 0, 0.2

# for x, y in test_loader:
#     pred = model(x)

#     if pred.argmax(1) != y:
#         print('\nThe model is not correct for this sample! Skip the sample!\n')
#         print('\n===========================\n')
#     else:
#         print('pred img = {}'.format(pred.detach().numpy().reshape(-1)))
#         print('lbl imp = {}\n'.format(y.item()))

#         if attack(model, x, y, eps): num_adv += 1
#         num_img += 1

#         print('\n===========================\n')
#         if num_img == 20:
#             print('Adv imgs = {}\n'.format(num_adv))
#             break

# def denormalize(x):
#     x = (x * 255).astype('uint8')
#     x = x.reshape(28,28)

#     return x


# def display(x, y, x_adv, y_adv):
#     x = denormalize(x)
#     x_adv = denormalize(x_adv)

#     fig, ax = plt.subplots(1, 2)

#     ax[0].set(title='Original. Label is {}'.format(y))
#     ax[1].set(title='Adv. sample. Label is {}'.format(y_adv))

#     ax[0].imshow(x, cmap='gray')
#     ax[1].imshow(x_adv, cmap='gray')
    
#     plt.show()