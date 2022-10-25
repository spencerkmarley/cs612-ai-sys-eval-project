
import torch
from torch import nn
from torch import linalg as LA
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import random
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from torch import Tensor
# #from typing import Any
# get_ipython().run_line_magic('matplotlib', 'inline')
import sys
sys.path.append('../')
import os 

# Load model definitions
import models
from models.definitions import MNISTNet, CIFAR10Net, CIFAR100Net

# import models.definitions.CIFAR100Net as CIFAR100
# import models.definitions.CIFAR10Net as CIFAR10
# import models.definitions.MNISTNet as MNIST

# Class names for CIFAR10
class_names_MNIST=['0','1','2','3','4','5','6','7','8','9']
class_names_CIFAR10 = ['airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']
class_names_CIFAR100 =['beaver',	'dolphin',	'otter',	'seal',	'whale',
'aquarium fish',	'flatfish',	'ray',	'shark',	'trout',
'orchids',	'poppies',	'roses',	'sunflowers',	'tulips',
'bottles',	'bowls',	'cans',	'cups',	'plates',
'apples',	'mushrooms',	'oranges',	'pears',	'sweet peppers',
'clock',	'computer keyboard',	'lamp',	'telephone',	'television',
'bed',	'chair',	'couch',	'table',	'wardrobe',
'bee',	'beetle',	'butterfly',	'caterpillar',	'cockroach',
'bear',	'leopard',	'lion',	'tiger',	'wolf',
'bridge',	'castle',	'house',	'road',	'skyscraper',
'cloud',	'forest',	'mountain',	'plain',	'sea',
'camel',	'cattle',	'chimpanzee',	'elephant',	'kangaroo',
'fox',	'porcupine',	'possum',	'raccoon',	'skunk',
'crab',	'lobster',	'snail',	'spider',	'worm',
'baby',	'boy',	'girl',	'man',	'woman',
'crocodile',	'dinosaur',	'lizard',	'snake',	'turtle',
'hamster',	'mouse',	'rabbit',	'shrew',	'squirrel',
'maple',	'oak',	'palm',	'pine',	'train',
'bicycle',	'bus',	'motorcycle',	'pickup truck',	'truck',
'lawn-mower',	'rocket',	'streetcar',	'tank',	'tractor']


model_map={'CIFAR10':CIFAR10Net, 'CIFAR100':CIFAR100Net, 'MNIST':MNISTNet}
# model_map={'CIFAR10':CIFAR10, 'CIFAR100':CIFAR100, 'MNIST':MNIST}
triggersize_map={'CIFAR10':32, 'CIFAR100':32, 'MNIST':28}
dim_map={'CIFAR10':3, 'CIFAR100':3, 'MNIST':1}
trigger_type_map={'CIFAR10':[1,2], 'CIFAR100':[1,2], 'MNIST':[2]}
class_names_map={'CIFAR10':class_names_CIFAR10, 'CIFAR100':class_names_CIFAR100, 'MNIST':class_names_MNIST}
epochs_map={'CIFAR10':4 ,'CIFAR100':3, 'MNIST':2}
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if not os.path.isdir('../backdoor_triggers'):
    os.makedirs('../backdoor_triggers')
lr=0.01


# Defining L norms

def l1_norm(x: Tensor, y: Tensor=0) -> Tensor:
    """ Compute the L1 norm between two tensors """
    res = torch.abs(x - y)
    return torch.sum(res)

def l2_norm(x, y=0):
    """ Compute the L2 norm between two tensors """
    res = torch.sum((x - y) ** 2)
    return torch.sqrt(res)

def linf_norm(x, y=0):
    """ Compute the L-inf norm between two tensors """
    res = torch.max(torch.abs(x - y))
    return res
def MAD_anomaly_index(X): #X is a list of numbers (L1, L2 etc)
    Xm = np.median(X)
    devs = X-Xm
    abs_devs=abs(devs)
    MAD = np.median(abs_devs)
    degree_of_anomaly = devs/MAD #<-2
    return degree_of_anomaly #,(degree_of_anomaly<-2).sum()


def save_model(model, name):
    torch.save(model.state_dict(), name)
def load_model(model_class, name):
    model = model_class()
    model.load_state_dict(torch.load(name))

    return model
def generate_trigger(model, dataloader, delta_0,loss_fn, optimizer, device, bdtype):
    #returns the trigger after this iteration
    #delta_0 is the input trigger after last iteration
    size = len(dataloader.dataset)
    model.train()
    delta=delta_0.detach().clone().requires_grad_() #detach may not be needed
    delta.retain_grad() #may not needed
    #print(delta.is_leaf)
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        assert delta.requires_grad, "Error: requires_grad is false"
        x_stamped=torch.add(x,delta) #from here delta is part of the graph
        pred = model(x_stamped)
        if bdtype=='MNIST':
            loss = loss_fn(pred, y) + l1_norm(delta)+(delta<0).type(torch.float32).sum()
        else:    
            loss = loss_fn(pred, y) +l1_norm(delta[0,:,:])+l1_norm(delta[0,:,:]-delta[1,:,:])+l1_norm(delta[0,:,:]-delta[2,:,:])+l1_norm(delta[1,:,:]-delta[2,:,:])
            #loss = loss_fn(pred, y) +LA.norm(LA.norm((torch.abs(delta)>0.01).type(torch.float32) ,2, dim=2),2)#+LA.norm(LA.norm((delta-0.5),1, dim=2),1)
        
        optimizer.zero_grad()         
        loss.backward(inputs=delta)#(retain_graph=True)
        #print(delta.grad.data.sum())
        #optimizer.step()
        temp = delta.detach().clone()
        delta=(temp-(delta.grad*lr)).requires_grad_()
        #delta.grad.data.zero_()
        if batch % 1000 == 0:
            #print(w_Trigger.is_leaf,w_Trigger.grad.data.sum())
            loss, current = loss.item(), batch * len(x)
            print('loss: {:.4f} [{}/{}]'.format(loss, current, size))
    return delta
def test_trigger(model, dataloader,delta, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.to(device)
    model.eval()
    loss, correct = 0.0, 0    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            x_stamped=torch.add(x,delta)
            pred = model(x_stamped)
            loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.int).sum().item()
    
    loss /= num_batches
    correct /= size
    print('Test Result: Accuracy @ {:.2f}%, Avg loss @ {:.4f}\n'.format(100 * correct, loss))
    return correct

def func_trigger_synthesis(MODELNAME,MODELCLASS,CLASSES,CIFAR100=True):
    """
    Example:
    MODELNAME='cifar10_backdoored_1' 
    MODELCLASS='CIFAR10' #'MNIST'
    CLASSES=[i for i in range(10)]
    default:optimise L1 norm of triggers
    ad-hoc:optimise L2 norm of triggers, to generate 'smoother' trigger patterns, i.e. invisible backdoor
    """
    if CIFAR100==False:
        CLASSES = list(range(0,10))
        
    TriggerSize=triggersize_map[MODELCLASS]
    testmodel=load_model(model_map[MODELCLASS],  f'../models/subject/{MODELNAME}.pt')
    testmodel=testmodel.to(device)
    transform = transforms.ToTensor()
    train_kwargs = {'batch_size': 100, 'shuffle':True}
    test_kwargs = {'batch_size': 1000}
    
    optimizer = optim.Adam(testmodel.parameters(), lr=0.1) # not using optimizer here
    num_of_epochs = epochs_map[MODELCLASS]
    # to map dataset, add this block if need to re-download a fresh dataset
    trainset_map={'CIFAR10':torchvision.datasets.CIFAR10(root='../data', train=True,download=True, transform=transform),
                  'CIFAR100':torchvision.datasets.CIFAR100(root='../data', train=True,download=True, transform=transform),
                  'MNIST':torchvision.datasets.MNIST(root='../data', train=True,download=True, transform=transform)}
    testset_map={'CIFAR10':torchvision.datasets.CIFAR10(root='../data', train=False,download=True, transform=transform),
                  'CIFAR100':torchvision.datasets.CIFAR100(root='../data', train=False,download=True, transform=transform),
                  'MNIST':torchvision.datasets.MNIST(root='../data', train=False,download=True, transform=transform)}


     # change to selected classes for CIFAR100 !!!
    triggers1={}
    acc1={}
    for TARGET in CLASSES:
        #initialize trigger to be 0.5
        delta =torch.zeros([dim_map[MODELCLASS],TriggerSize,TriggerSize], requires_grad=True, device=device)+0.5
        trainset = trainset_map[MODELCLASS]
        testset = testset_map[MODELCLASS]

        for i in range(len(trainset)):
            trainset.targets[i]=TARGET  
        for i in range(len(testset)):
            testset.targets[i]=TARGET  

        trigger_gen_loader = DataLoader(trainset, **train_kwargs)
        trigger_test_loader = DataLoader(testset, **test_kwargs)

        for epoch in range(num_of_epochs):
            print(f'With target number {TARGET}:' )
            delta=generate_trigger(testmodel, trigger_gen_loader, delta , nn.CrossEntropyLoss(), optimizer, device, bdtype=MODELCLASS)
            test_acc=test_trigger(testmodel, trigger_test_loader,delta, nn.CrossEntropyLoss(), device)
        triggers1[TARGET]=delta
        acc1[TARGET]=test_acc
    
    
# print(  "".join("{:10.2f}".format(l1_norm(triggers1[i]).item()) for i in range(10))  )
# print(  "".join("{:10.2f}".format(l2_norm(triggers1[i]).item()) for i in range(10))  )
# print(  "".join("{:10.2f}".format(linf_norm(triggers1[i]).item()) for i in range(10))  )
# print("      "+"      ".join([str((abs(triggers1[i])>0.05).sum().item()) for i in range(10)]))
# print("".join(["{:10.4f}".format(value) for key,value in acc1.items()]))
    
    l1_anom= MAD_anomaly_index([l1_norm(triggers1[i]).item() for i in CLASSES]) <-2  
    acc_anom=MAD_anomaly_index([value for key,value in acc1.items()]) >2
    #outliers according to L1 norms of triggers
    outliers=[ CLASSES[i] for i in range(len(CLASSES)) if l1_anom[i] ]
    #outliers according to attack accuracy - backdoored classes usually have higher accuracy 
    acc_outliers=[CLASSES[i] for i in range(len(CLASSES)) if acc_anom[i] and acc1[CLASSES[i]]>0.5]

    if  len(outliers)==0:
        outliers.extend(acc_outliers)
    if  len(outliers)>0:
        for i in outliers:
            torch.save(triggers1[i],f"../backdoor_triggers/{MODELNAME}_class_{i}.pt")
        print("Infected Classes: ",outliers)

        print("Infected Classes Names: "+" ".join(( class_names_map[MODELCLASS][i] for i in outliers)))
        print("trigger saved in folder ../backdoor_triggers")
        #How to load: t=torch.load( f"backdoor_triggers/{MODELNAME}_class_{i}.pt",map_location=torch.device('cpu')).detach().numpy()
    return outliers, acc1

if __name__ == '__main__':
    #cifar_backdoored_classes = func_trigger_synthesis('cifar10_backdoored_1','CIFAR10',[i for i in range(10)] )[0]
    mnist_backdoored_classes = func_trigger_synthesis('mnist_backdoored_1','MNIST',[i for i in range(10)] )[0]
    # run ad-hoc 
        


