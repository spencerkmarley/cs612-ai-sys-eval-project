# Import all standard libraries used in course 

import random
import math

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

from PIL import ImageDraw, ImageFont

# For reproducibility
torch.manual_seed(42)

"""
Test the robustness of a model by:
(i) taking clean samples
(ii) perturbing them by a perturbation that we know doesn't change the label
(iii) using the subject model to classify the pertubred images
(iv) concluding that there is a backdoor if we discover discrepancies
"""

def test_robust(benign, subject, dataset, test, num_img, eps, threshold, verbose=False):

    robust = True

    if test == 0:
        perturb_rotation(benign, subject, dataset, test, num_img, eps, threshold, verbose=False)
    elif test == 1:
        perturb_change_pixels(benign, subject, dataset, test, num_img, eps, threshold, verbose=False)
    elif test == 2:
        perturb_invert(benign, subject, dataset, test, num_img, eps, threshold, verbose=False)
    elif test == 3:
        perturb_change_lighting(benign, subject, dataset, test, num_img, eps, threshold, verbose=False)
    elif test == 4:
        perturb_zoom_in_out(benign, subject, dataset, test, num_img, eps, threshold, verbose=False)
    elif test == 5:
        perturb_resize(benign, subject, dataset, test, num_img, eps, threshold, verbose=False)
    elif test == 6:
        perturb_crop_rescale(benign, subject, dataset, test, num_img, eps, threshold, verbose=False)
    elif test == 7:
        perturb_bit_depth_reduction(benign, subject, dataset, test, num_img, eps, threshold, verbose=False)
    elif test == 8:
        perturb_compress_decompress(benign, subject, dataset, test, num_img, eps, threshold, verbose=False)
    elif test == 9:
        perturb_total_var_min(benign, subject, dataset, test, num_img, eps, threshold, verbose=False)
    elif test == 10:
        perturb_adding_noise(benign, subject, dataset, test, num_img, eps, threshold, verbose=False)
    elif test == 11:
        perturb_watermark(benign, subject, dataset, test, num_img, eps, threshold, verbose=False)
    elif test == 12:
        perturb_whitesquare(benign, subject, dataset, test, num_img, eps, threshold, verbose=False)
    else:
        print("Please provide a valid test number")
    
    return robust


def perturb_rotation(benign, subject, dataset, test, num_img, eps, threshold, verbose=False):
    '''
    Randomly sample 20% of the test images for perturbation by rotation.
    The rotation range is set to between 45-60 degrees.
    <By Titus>
    '''
    # Perturb some clean samples by rotating them
    if verbose:
        print("Perturbing by rotation...")
    
    robust = True
    rotate = RandomRotation(degrees=(45, 60))
    
    #We sample images amounting to 20% of the dataset and rotate them
    indices_to_rotate = random.sample(range(num_img), math.ceil(num_img*0.2))
    
    test_loader = torch.utils.data.DataLoader(dataset, batch_size = 1)
    discrepancies = 0
    
    for batch, (x, y) in enumerate(test_loader):
        if batch in indices_to_rotate:
            x_rotate = rotate(x)
            prediction_benign, prediction_subject = benign(x), subject(x)
            prediction_rotated_benign, prediction_rotated_subject = benign(x_rotate), subject(x_rotate)
            
            #if the subject model predicts differently on rotation, discrepancies +=1 if the benign model predicts differently as well
            if prediction_rotated_subject.argmax(1)!=prediction_subject.argmax(1):
                if prediction_rotated_benign.argmax(1)==prediction_benign.argmax(1):
                    discrepancies+=1
        
    if discrepancies/len(indices_to_rotate)>= threshold:
        robust = False  
    
    if verbose:
        print("Discrepancy = {} %\n".format(100*discrepancies/len(indices_to_rotate)))
        if robust:
            print("Model is robust")
        else:
            print("Model is not robust")
    
    return robust

def perturb_change_pixels(benign, subject, dataset, test, num_img, eps, threshold, verbose=False):
    """
    Perturb some clean samples by changing pixels
    """

    robust = True

    # Take clean samples
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    count = 0
    discrepancies = 0
    num_perturbed = 0

    # Perturb them by a perturbation that doesn't change the label
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
            
            # Check if the label stays the same
            prediction_benign = benign(x_perturb)
            if prediction.argmax(1) == prediction_benign.argmax(1):
                same_label = True
                num_perturbed += 1

            # Use subject model to classify the pertubred images
            if same_label:
                prediction_subject = subject(x_perturb)
                
                if prediction_benign.argmax(1) != prediction_subject.argmax(1):
                    discrepancies += 1

            count += 1

    # Conclude that there is a backdoor if we discover discrepancies    
    if discrepancies/num_perturbed >= threshold:
        robust = False

    if verbose:
        print("Discrepancy = {} %\n".format(100*discrepancies/num_perturbed))
        if robust:
            print("Model is robust")
        else:
            print("Model is not robust")
    
    return robust

def perturb_invert(benign, subject, dataset, test, num_img, eps, threshold, verbose=False):
    '''
    Randomly sample 20% of images for color inversion.
    <By Titus>
    '''
    # Perturb some clean samples by inverting them
    if verbose:
        print("Perturbing by inverting images...")
    
    robust = True
    
    #We sample images amounting to 20% of the dataset and rotate them
    indices_to_invert = random.sample(range(num_img), math.ceil(num_img*0.2))
    
    test_loader = torch.utils.data.DataLoader(dataset, batch_size = 1)
    discrepancies = 0
    
    for batch, (x, y) in enumerate(test_loader):
        if batch in indices_to_invert:
            x_invert = invert(x)
            prediction_benign, prediction_subject = benign(x), subject(x)
            prediction_invert_benign, prediction_invert_subject = benign(x_invert), subject(x_invert)
            
            #if the subject model predicts differently on rotation, discrepancies +=1 if the benign model predicts differently as well
            if prediction_invert_subject.argmax(1)!=prediction_subject.argmax(1):
                if prediction_invert_benign.argmax(1)==prediction_benign.argmax(1):
                    discrepancies+=1
                    if verbose:
                        plt.imshow(x_invert.permute(1,2,0))
                        plt.title(f'Rotated image of class {y} predicted to be class {prediction_invert_subject.argmax(1)}')
        
    if discrepancies/len(indices_to_invert)>= threshold:
        robust = False  
   
    if verbose:
        print("Discrepancy = {} %\n".format(100*discrepancies/len(indices_to_invert)))
        if robust:
            print("Model is robust")
        else:
            print("Model is not robust")  
    
    return robust

def perturb_change_lighting(benign, subject, dataset, test, num_img, eps, threshold, verbose=False):
    '''
    Perturb 20% of clean samples by changing the lighting
    <By Titus>
    '''
    if verbose:
        print("Perturbing by changing the lighting...")
    robust = True
    
    #We sample images amounting to 20% of the dataset and rotate them
    indices = random.sample(range(num_img), math.ceil(num_img*0.2))
    
    test_loader = torch.utils.data.DataLoader(dataset, batch_size = 1)
    discrepancies = 0
    
    for batch, (x, y) in enumerate(test_loader):
        same_label = False
        if batch in indices:
            x_bright = adjust_brightness(x,4.0)
            prediction_benign, prediction_subject = benign(x), subject(x)
            prediction_bright_benign, prediction_bright_subject = benign(x_bright), subject(x_bright)
            
            #if the subject model predicts differently on rotation, discrepancies +=1 if the benign model predicts differently as well
            if prediction_bright_subject.argmax(1)!=prediction_subject.argmax(1):
                if prediction_bright_benign.argmax(1)==prediction_benign.argmax(1):
                    discrepancies+=1
                    if verbose:
                        plt.imshow(x_bright.permute(1,2,0))
                        plt.title(f'Rotated image of class {y} predicted to be class {prediction_bright_subject.argmax(1)}')
        
    if discrepancies/len(indices)>= threshold:
        robust = False  
    
    if verbose:
        print("Discrepancy = {} %\n".format(100*discrepancies/len(indices)))
        if robust:
            print("Model is robust")
        else:
            print("Model is not robust")       
    
    return robust

def perturb_zoom_in_out(benign, subject, dataset, test, num_img, eps, threshold, verbose=False):
    '''
    Perturb 20% of clean samples by zooming in and out. 
    References: https://stackoverflow.com/questions/64727718/clever-image-augmentation-random-zoom-out
    
    <By Titus>
    '''
    if verbose:
        print("Perturbing by zooming in and out...")
    robust = True
    
    #We sample images amounting to 20% of the dataset and rotate them
    indices = random.sample(range(num_img), math.ceil(num_img*0.2))
    Crop = RandomResizedCrop((28,28),(0.2,0.8))
    
    test_loader = torch.utils.data.DataLoader(dataset, batch_size = 1)
    _,_,shape = (next(iter(test_loader)))[0][0].shape
    crop = RandomResizedCrop((shape,shape),(0.2,0.8))
    discrepancies = 0
    
    for batch, (x, y) in enumerate(test_loader):
        if batch in indices:
            x_zoom = crop(x)
            prediction_benign, prediction_subject = benign(x), subject(x)
            prediction_zoom_benign, prediction_zoom_subject = benign(x_zoom), subject(x_zoom)
            
            #if the subject model predicts differently on rotation, discrepancies +=1 if the benign model predicts differently as well
            if prediction_zoom_subject.argmax(1)!=prediction_subject.argmax(1):
                if prediction_zoom_benign.argmax(1)==prediction_benign.argmax(1):
                    discrepancies+=1
                    if verbose:
                        plt.imshow(x_zoom.permute(1,2,0))
                        plt.title(f'Rotated image of class {y} predicted to be class {prediction_zoom_subject.argmax(1)}')
        
    if discrepancies/len(indices)>= threshold:
        robust = False  
    
    if verbose:
        print("Discrepancy = {} %\n".format(100*discrepancies/len(indices)))
        if robust:
            print("Model is robust")
        else:
            print("Model is not robust")      
    
    return robust

def perturb_resize(benign, subject, dataset, test, num_img, eps, threshold, verbose=False):
    '''
    Titus: This function might be a problem because the networks are trained to take in 
    image of a specific dimension right?
    '''
    # Perturb some clean samples by resizing
    if verbose:
        print("Perturbing by resizing...")
    return dataset

def perturb_crop_rescale(benign, subject, dataset, test, num_img, eps, threshold, verbose=False):
    '''
    Perturb 20% of clean samples by cropping and rescaling
    
    <By Titus>
    '''
    # Perturb some clean samples by cropping and rescaling
    if verbose:
        print("Perturbing by cropping and rescaling...")
    robust = True
    
    #We sample images amounting to 20% of the dataset and rotate them
    indices = random.sample(range(num_img), math.ceil(num_img*0.2))
    
    test_loader = torch.utils.data.DataLoader(dataset, batch_size = 1)
    _, _, shape = next(iter(test_loader))[0][0].shape
    
    cropper = RandomCrop(size = (16,16))
    resize = Resize((shape,shape))
    
    discrepancies = 0
    
    for batch, (x, y) in enumerate(test_loader):
        if batch in indices:
            x_crop = resize(cropper(x))
            prediction_benign, prediction_subject = benign(x), subject(x)
            prediction_crop_benign, prediction_crop_subject = benign(x_crop), subject(x_crop)
            
            #if the subject model predicts differently on rotation, discrepancies +=1 if the benign model predicts differently as well
            if prediction_crop_subject.argmax(1)!=prediction_subject.argmax(1):
                if prediction_crop_benign.argmax(1)==prediction_benign.argmax(1):
                    discrepancies+=1
                    if verbose:
                        plt.imshow(x_crop.permute(1,2,0))
                        plt.title(f'Rotated image of class {y} predicted to be class {prediction_crop_subject.argmax(1)}')
        
    if discrepancies/len(indices)>= threshold:
        robust = False  
    
    if verbose:
        print("Discrepancy = {} %\n".format(100*discrepancies/len(indices)))
        if robust:
            print("Model is robust")
        else:
            print("Model is not robust")       
    
    return robust

def perturb_bit_depth_reduction(benign, subject, dataset, test, num_img, eps, threshold, verbose=False):
    '''
    Perturb 20% of clean samples by bitwise depth reduction.
    
    <By Titus>
    '''
    # Perturb some clean samples by bit depth reduction
    if verbose:
        print("Perturbing by bit depth reduction...")
    robust = True
    
    #We sample images amounting to 20% of the dataset and rotate them
    indices = random.sample(range(num_img), math.ceil(num_img*0.2))   
    test_loader = torch.utils.data.DataLoader(dataset, batch_size = 1)

    discrepancies = 0
    
    for batch, (x, y) in enumerate(test_loader):
        if batch in indices:
            sourceTensor = (x*255).clone()
            # sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True)
            # torch.tensor(sourceTensor)

            c = torch.tensor(sourceTensor, dtype = torch.uint8) #necessary for posterize function
            posterizer = RandomPosterize(bits=2)
            c_pos = posterizer(c)
            x_pos = torch.div(c_pos, 255) #Must normalize, but this converts dtype back to float tensor.
            prediction_benign, prediction_subject = benign(x), subject(x)
            prediction_pos_benign, prediction_pos_subject = benign(x_pos), subject(x_pos)
            
            #if the subject model predicts differently on rotation, discrepancies +=1 if the benign model predicts differently as well
            if prediction_pos_subject.argmax(1)!=prediction_subject.argmax(1):
                if prediction_pos_benign.argmax(1)==prediction_benign.argmax(1):
                    discrepancies+=1
                    if verbose:
                        plt.imshow(c_pos.permute(1,2,0))
                        plt.title(f'Rotated image of class {y} predicted to be class {prediction_pos_subject.argmax(1)}')
        
    if discrepancies/len(indices)>= threshold:
        robust = False  
    
    if verbose:
        print("Discrepancy = {} %\n".format(100*discrepancies/len(indices)))
        if robust:
            print("Model is robust")
        else:
            print("Model is not robust")       
    
    return robust

def perturb_compress_decompress(benign, subject, dataset, test, num_img, eps, threshold, verbose=False):
    '''
    Titus: Is this what you mean by compress? https://www.geeksforgeeks.org/how-to-compress-images-using-python-and-pil/
    '''
    # Perturb some clean samples by compressing and decompressing
    if verbose:
        print("Perturbing by compressing and decompressing...")
    return dataset

def perturb_total_var_min(benign, subject, dataset, test, num_img, eps, threshold, verbose=False):
    # Perturb some clean samples by total var min
    if verbose:
        print("Perturbing by total var min...")
    return dataset

class AddGaussianNoise(object):
    '''
    Custom class to add Gaussian noise to images (tensors).
    <By Titus>
    '''
    def __init__(self, mean=0., std=0.5):
        '''
        The std was set to 0.5 instead of 1.0 because the resultant gaussed image using 
        an std of 1.0 was unrecognizable even to humans.
        '''
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def perturb_adding_noise(benign, subject, dataset, test, num_img, eps, threshold, verbose=False):
    '''
    Perturb 20% of clean samples clean samples by adding noise
    Uses the custom AddGaussianNoise class
    
    <By Titus>
    '''
    if verbose:
        print("Perturbing by adding noise...")
    robust = True
    
    #We sample images amounting to 20% of the dataset and rotate them
    indices = random.sample(range(num_img), math.ceil(num_img*0.2))   
    test_loader = torch.utils.data.DataLoader(dataset, batch_size = 1)
    Gauss = AddGaussianNoise(0,1)

    discrepancies = 0
    
    for batch, (x, y) in enumerate(test_loader):
        if batch in indices:
            x_gauss = Gauss(x)
            prediction_benign, prediction_subject = benign(x), subject(x)
            prediction_gauss_benign, prediction_gauss_subject = benign(x_gauss), subject(x_gauss)
            
            #if the subject model predicts differently on rotation, discrepancies +=1 if the benign model predicts differently as well
            if prediction_gauss_subject.argmax(1)!=prediction_subject.argmax(1):
                if prediction_gauss_benign.argmax(1)==prediction_benign.argmax(1):
                    discrepancies+=1
                    if verbose:
                        plt.imshow(x_gauss.permute(1,2,0))
                        plt.title(f'Rotated image of class {y} predicted to be class {prediction_gauss_subject.argmax(1)}')
        
    if discrepancies/len(indices)>= threshold:
        robust = False  
    
    if verbose:
        print("Discrepancy = {} %\n".format(100*discrepancies/len(indices)))
        if robust:
            print("Model is robust")
        else:
            print("Model is not robust")

    return robust

def perturb_watermark(benign, subject, dataset, test, num_img, eps, threshold, verbose=False):
    '''
    Add watermark to 20% of the test samples
    
    There's a bug here at .convert('RGBA') which turns the entire image to black and white.
    Not sure why that is. If this persists on your computer as well, maybe just drop this test.
    
    <By Titus>
    '''
    if verbose:
        print("Perturbing by adding a watermark...")
    robust = True
    
    #We sample images amounting to 20% of the dataset and rotate them
    indices = random.sample(range(num_img), math.ceil(num_img*0.2))
    
    test_loader = torch.utils.data.DataLoader(dataset, batch_size = 1)
    font = ImageFont.truetype("/Library/fonts/Arial.ttf", 5)
    discrepancies = 0
    
    for batch, (x, y) in enumerate(test_loader):
        if batch in indices:
            x_w = ToPILImage()(x.clone().data).convert('RGBA')
            draw = ImageDraw.Draw(x_w)
            draw.text((0, 0), "TADA", (255, 255, 255), font=font)
            x_w = pil_to_tensor(x_w)
            x_w = x_w/255
            prediction_benign, prediction_subject = benign(x), subject(x)
            prediction_watermark_benign, prediction_watermark_subject = benign(x_w), subject(x_w)
            
            #if the subject model predicts differently on rotation, discrepancies +=1 if the benign model predicts differently as well
            if prediction_watermark_subject.argmax(1)!=prediction_subject.argmax(1):
                if prediction_watermark_benign.argmax(1)==prediction_benign.argmax(1):
                    discrepancies+=1
        
    if discrepancies/len(indices)>= threshold:
        robust = False  
    
    if verbose:
        print("Discrepancy = {} %\n".format(100*discrepancies/len(indices)))
        if robust:
            print("Model is robust")
        else:
            print("Model is not robust")      
    
    return robust

def perturb_whitesquare(benign, subject, dataset, test, num_img, eps, threshold, verbose=False):
    '''
    Perturb 20% of clean samples by adding a white square
    <By Titus>
    '''
    if verbose:
        print("Perturbing by adding white square...")  
    
    robust = True
    
    #We sample images amounting to 20% of the dataset and rotate them
    indices = random.sample(range(num_img), math.ceil(num_img*0.2))
    
    test_loader = torch.utils.data.DataLoader(dataset, batch_size = 1)
    discrepancies = 0
    
    for batch, (x, y) in enumerate(test_loader):
        if batch in indices:
            x_sq = ToPILImage()((x*255).clone().data).convert('RGBA')
            draw = ImageDraw.Draw(x_sq)
            draw.rectangle((0, 0, 3, 3), fill=(255, 255, 255))
            x_sq = pil_to_tensor(x_sq)
            x_sq = torch.div(x_sq, 255.0) #must renormalize this.
            prediction_benign, prediction_subject = benign(x), subject(x)
            prediction_sq_benign, prediction_sq_subject = benign(x_sq), subject(x_sq)
            
            #if the subject model predicts differently on rotation, discrepancies +=1 if the benign model predicts differently as well
            if prediction_sq_subject.argmax(1)!=prediction_subject.argmax(1):
                if prediction_sq_benign.argmax(1)==prediction_benign.argmax(1):
                    discrepancies+=1
        
    if discrepancies/len(indices)>= threshold:
        robust = False  
    
    if verbose:
        print("Discrepancy = {} %\n".format(100*discrepancies/len(indices)))
        if robust:
            print("Model is robust")
        else:
            print("Model is not robust")     
    
    return robust
