#changes :
#added learning rate auto-adjustment in GD
#running badnet triggers as well as invisible triggers, print result for both
#saving ONLY backdoor triggers for visualization
#incorporate logging 

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from util import get_pytorch_device, logger

import torchvision
from torchvision.transforms import ToTensor
from torchvision import datasets, transforms

# Utility functions
from util import config as c

import numpy as np
import math
import random
import os
import sys
sys.path.append('../')

# Load model definitions
from models.definitions import MNISTNet, CIFAR10Net, CIFAR100Net

# Class names
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
triggersize_map={'CIFAR10':32, 'CIFAR100':32, 'MNIST':28}
dim_map={'CIFAR10':3, 'CIFAR100':3, 'MNIST':1}
trigger_type_map={'CIFAR10':[1,2], 'CIFAR100':[1,2], 'MNIST':[2]}
class_names_map={'CIFAR10':class_names_CIFAR10, 'CIFAR100':class_names_CIFAR100, 'MNIST':class_names_MNIST}
epochs_map={'CIFAR10':4 ,'CIFAR100':3, 'MNIST':2}
device = get_pytorch_device()

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
    model.load_state_dict(torch.load(name, map_location=device))

    return model

def generate_trigger(model, dataloader, delta_0,loss_fn, optimizer, device, bdtype,lr):
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
        if bdtype=='MNIST': # for mnist use simpler loss is sufficient
            loss = loss_fn(pred, y) + l1_norm(delta)+(delta<0).type(torch.float32).sum()
        elif bdtype=='CIFAR': # CIFAR data, assuming badnet - small pattern backdoor   
            loss = loss_fn(pred, y) +l1_norm(delta[0,:,:])+\
                l1_norm(torch.maximum(torch.maximum(delta[0,:,:],delta[1,:,:]),delta[2,:,:] )
                        -torch.minimum(torch.minimum(delta[0,:,:],delta[1,:,:]),delta[2,:,:]))
        else:  # CIFAR data, assuming invisible backdoor  - large but low intensity pattern
            loss = loss_fn(pred, y) +l2_norm(delta)
                    
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
    
def func_trigger_synthesis(MODELNAME, MODELCLASS, TRIGGERS, CIFAR100_PCT=1, CLASSES=None):
    """
    Example:
    MODELNAME='cifar10_backdoored_1' 
    MODELCLASS='CIFAR10' #'MNIST'
    CLASSES=[i for i in range(10)]
    CIFAR100_PCT: float in [0.04 , 1]
    1st round:optimise L1 norm of triggers, to generate badnet backdoors
    2nd round:optimise L2 norm of triggers, to generate 'smoother' trigger patterns, i.e. invisible backdoor
    """
    
    if CLASSES==None:
        CLASSES = list(range(0,100)) if MODELCLASS=='CIFAR100' else list(range(0,10))
    #print(CLASSES) 
    #logger.info('WE FOUND A BACKDOOR BAYBAY')
    
    TriggerSize=triggersize_map[MODELCLASS]
    testmodel=load_model(model_map[MODELCLASS],  "./" + MODELNAME)
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
    
    sample_indexes = random.sample(range(50000), math.floor(50000*CIFAR100_PCT)) # this is for cifar100 so hardcoded
    trainset = trainset_map[MODELCLASS]
    testset = testset_map[MODELCLASS]
    
    if MODELCLASS=='CIFAR100':#use reduced dataset
        trainset.data = trainset.data[sample_indexes]
        trainset.targets = list(Tensor(trainset.targets)[sample_indexes])
        
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
        acc_now=0
        lr0=lr
        for epoch in range(num_of_epochs):
            print(f'With target number {TARGET}:' )
            delta=generate_trigger(testmodel, trigger_gen_loader, delta , nn.CrossEntropyLoss(), optimizer, device, bdtype=MODELCLASS[:5], lr=lr0)
            test_acc=test_trigger(testmodel, trigger_test_loader,delta, nn.CrossEntropyLoss(), device)
                #LR adjustment
            if test_acc<acc_now-0.005:
                lr0=lr0/10
                print(f"learning rate start:{lr} -> learning rate now:{lr0}")
            acc_now= test_acc              
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
    outliers=[ CLASSES[i] for i in range(len(CLASSES)) if l1_anom[i] and acc1[CLASSES[i]]>0.6]
    #outliers according to attack accuracy - backdoored classes usually have higher accuracy 
    acc_outliers=[CLASSES[i] for i in range(len(CLASSES)) if acc_anom[i] and acc1[CLASSES[i]]>0.6]
    
    if not os.path.isdir(TRIGGERS):
        os.makedirs(TRIGGERS)

    if  len(outliers)==0:
        outliers.extend(acc_outliers)
    if  len(outliers)>0: #found outliers
        print("Finding badnet triggers... ")
        print("Infected Classes: ", outliers)

        print("Infected Classes Names: "+" ".join(( class_names_map[MODELCLASS][i] for i in outliers)))
    else:
        print("Did not find badnet backdoor")
    txt= MODELNAME+" badnet backdoors classes: "+" ".join(str(i) for i in outliers)
    logger.info(txt)
    txt= MODELNAME+" badnet backdoors: "+" ".join(class_names_map[MODELCLASS][i] for i in outliers)
    logger.info(txt)
    for i in outliers:
        torch.save(triggers1[i], TRIGGERS + f"/class_{i}_bn.pt")
        
    print("\n")
#*********************************************new******************************************************        
    del delta    # just in case..
    triggers2={}
    acc2={}
    differs=[]
    
    if MODELCLASS!='MNIST':
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
            acc_now=0
            lr0=lr
            for epoch in range(num_of_epochs):
                print(f'With target number {TARGET}:' )
                delta=generate_trigger(testmodel, trigger_gen_loader, delta , nn.CrossEntropyLoss(), optimizer, device, bdtype='OTHER', lr=lr0)
                test_acc=test_trigger(testmodel, trigger_test_loader,delta, nn.CrossEntropyLoss(), device)
                #LR adjustment
                if test_acc<acc_now-0.005:
                    lr0=lr0/10
                    print(f"learning rate start:{lr} -> learning rate now:{lr0}")
                acc_now= test_acc       
            triggers2[TARGET]=delta
            acc2[TARGET]=test_acc # not using accuracy as L2 norm can lead to very good accuracy optimization on any model
            if l2_norm(delta).item()<0.0078*TriggerSize**2 or linf_norm(delta).item()<0.25: # imperical thresthold used here
                differs.append(TARGET)
        if  len(differs)>0: 
            print("Finding invisible triggers... ")
            print("Infected Classes: ", differs)

            print("Infected Classes Names: "+" ".join(( class_names_map[MODELCLASS][i] for i in differs)))
        else:
            print("Did not find invisible backdoor")
        for i in differs:
            torch.save(triggers2[i], TRIGGERS + f"/class_{i}_iv.pt")
        
    print("triggers saved in folder " + TRIGGERS)

    #How to load: t=torch.load(TRIGGERS + f"_class_{i}.pt", map_location=torch.device('cpu')).detach().numpy()    
        
#*********************************************new******************************************************         
       
    txt= MODELNAME+" invisible backdoor classes: "+" ".join(str(i) for i in differs)
    logger.info(txt)
    txt= MODELNAME+" invisible backdoors: "+" ".join(class_names_map[MODELCLASS][i] for i in differs)
    logger.info(txt)    
    return outliers,differs

if __name__ == '__main__':
    #outliers,differs= func_trigger_synthesis(MODELNAME="../models/subject/mnist_backdoored_1.pt", MODELCLASS='MNIST', TRIGGERS="./backdoor_triggers/mnist_backdoored_1/")
    outliers,differs=cifar_10_backdoored_classes = func_trigger_synthesis(MODELNAME="./models/subject/best_model_CIFAR10_10BD.pt", MODELCLASS='CIFAR10', TRIGGERS="./backdoor_triggers/best_model_CIFAR10_10BD/")
    #outliers,differs=cifar_100_backdoored_classes = func_trigger_synthesis(MODELNAME="../models/subject/CIFAR100_bn_BD5.pt", MODELCLASS='CIFAR100', TRIGGERS="./backdoor_triggers/CIFAR100_bn_BD5/", CIFAR100_PCT=0.04)
