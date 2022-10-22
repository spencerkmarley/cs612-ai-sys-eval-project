# Import standard libraries

import random

import torch

from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, RandomRotation, RandomResizedCrop, ToPILImage, RandomCrop, Resize, RandomPosterize
from torchvision.transforms.functional import invert, adjust_brightness, pil_to_tensor, posterize

import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

from copy import deepcopy

from PIL import ImageFont, ImageDraw

import warnings
warnings.filterwarnings("ignore")

def denormalize(tensor):
    return tensor*255.

transform = transforms.ToTensor()
train_kwargs = {'batch_size': 10, 'shuffle':True}
trainset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, **train_kwargs)

device = 'cpu'
for batch, (x,y) in enumerate(train_loader):
    x,y = x.to(device), y.to(device)
    break


## Rotation
rotate = RandomRotation(degrees=(45,60))
indices_to_rotate = random.sample(range(len(x)),2)

# plt.imshow(x[0].permute(1,2,0))

for i in range(len(x)):
    if i in indices_to_rotate:
        # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
        # axes[0].imshow(x[i].permute(1,2,0))
        # axes[0].set(title = 'Original')
        x[i] = rotate(x[i])
        # axes[1].imshow(x[i].permute(1,2,0))
        # axes[1].set(title = 'Rotated')

## Inversion
# plt.imshow(invert(x[indices_to_rotate[0]]).permute(1,2,0))
# plt.title('Inverted image');

## Lighting change
# plt.imshow(adjust_brightness(x[indices_to_rotate[0]],2.0).permute(1,2,0))

## Zooming in
_,_,shape = next(iter(train_loader))[0][0].shape
# shape

Crop = RandomResizedCrop((shape,shape),(0.2,0.8))
# plt.imshow(Crop(x[indices_to_rotate[0]]).permute(1,2,0))

## Random cropping
cropper = RandomCrop(size = (16,16))
resize = Resize((shape,shape))
# plt.imshow(resize(cropper(x[indices_to_rotate[0]])).permute(1,2,0))

## Bitwise depth reduction
c1 = torch.tensor((x[0]*255).clone(), dtype = torch.uint8)
posterizer = RandomPosterize(bits=2)
# fig, axes = plt.subplots(nrows=1, ncols=2)
# axes[0].imshow(x[0].permute(1,2,0))
# axes[0].set(title='Original image')
# axes[1].imshow(posterizer(c1).permute(1,2,0))
# axes[1].set(title='Feature reduced image')


# class AddGaussianNoise(object):
#     def __init__(self, mean=0., std=0.5):
#         '''
#         Reduced the std from 1 to 0.5 because the resultant gaussed image was unrecognizable to humans.
#         '''
#         self.std = std
#         self.mean = mean
        
#     def __call__(self, tensor):
#         return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
#     def __repr__(self):
#         return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

# Gauss = AddGaussianNoise(0,0.5)
# Gauss

# plt.imshow(Gauss(x[indices_to_rotate[0]]).permute(1,2,0))

# x_w = x_w/255

# plt.imshow(x_w.permute(1,2,0))

# from PIL import ImageFont, ImageDraw

# x_w = x[indices_to_rotate[0]].clone()
# x_w = ToPILImage()(x[indices_to_rotate[0]].clone().data).convert('RGBA')
# draw = ImageDraw.Draw(x_w)
# font = ImageFont.truetype("/Library/fonts/Arial.ttf", 5)

# draw.text((0, 0), "TADA", 
#           (255, 255, 255), font=font)
# x_w = pil_to_tensor(x_w)
# plt.title("White text")
# plt.imshow(x_w.permute(1,2,0))

# for i in range(len(x)):
#     if i in indices_to_rotate:
#         img = ToPILImage()(x[i].clone().data).convert('RGBA')
#         draw = ImageDraw.Draw(img)
#         draw.rectangle((0, 0, 3, 3), fill=(255, 255, 255))
#         img = pil_to_tensor(img)
#         img = torch.div(img, 255.0)
#         fig,axes = plt.subplots(nrows=1,ncols=2)
#         axes[0].imshow(x[i].permute(1,2,0),cmap='gray')
#         axes[1].imshow(img.permute(1,2,0))