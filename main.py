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

# Import custom libraries
from model_tests import robustness as rb
from model_tests import test_retrain_weights as rt
from model_tests import backdoor_forget as bd
from model_tests import trigger_synthesis as tss
from models.train import train_mnist

# Load model definitions
import models
from models.definitions import MNISTNet, CIFAR10Net, CIFAR100Net
from util import get_pytorch_device

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

device = get_pytorch_device()

# Provide filepaths
data_file_path = "data/"

# Provide test cases
TEST_CASE = 3

if TEST_CASE == 1:
    model_string = "MNIST"
    network_definition = MNISTNet()
    benign_model_file_path = "models/benign/mnist.pt"
    subject_model_file_path = "models/subject/mnist_backdoored_1.pt"
    retrained_model_file_path = "./models/retrained/retrained_mnist_"
    triggers = "./backdoor_triggers/mnist_backdoored_1/"
    trainset = datasets.MNIST(data_file_path, train=True, download=True, transform=transforms.ToTensor())
    testset = datasets.MNIST(data_file_path, train=False, download=True, transform=transforms.ToTensor())
    mnist = True
    CIFAR100 = False

elif TEST_CASE == 2:
    model_string = "CIFAR10"
    network_definition = CIFAR10Net()
    benign_model_file_path = "models/benign/benign_CIFAR10.pt"
    subject_model_file_path = "models/subject/best_model_CIFAR10_10BD.pt"
    retrained_model_file_path = "./models/retrained/retrained_CIFAR10_10BD_"
    triggers = "./backdoor_triggers/best_model_CIFAR10_10BD/"
    trainset = datasets.CIFAR10(data_file_path, train=True, download=True, transform=transforms.ToTensor())
    testset = datasets.CIFAR10(data_file_path, train=False, download=True, transform=transforms.ToTensor())
    mnist = False
    CIFAR100 = False

elif TEST_CASE == 3:
    model_string = "CIFAR100"
    network_definition = CIFAR100Net()
    benign_model_file_path = "models/benign/CIFAR100_seed3.pt"
    subject_model_file_path = "models/subject/CIFAR100_bn_BD5.pt"
    retrained_model_file_path = "./models/retrained/retrained_CIFAR100_"
    triggers = "./backdoor_triggers/CIFAR100_bn_BD5/"
    trainset = datasets.CIFAR100(data_file_path, train=True, download=True, transform=transforms.ToTensor())
    testset = datasets.CIFAR100(data_file_path, train=False, download=True, transform=transforms.ToTensor())
    mnist = False
    CIFAR100 = True

# Set parameters
NUM_IMG = 10000 #number of images in test set
EPS = 0.2
THRESHOLD = 0.1
N_CONTROL_MODELS = 2
VERBOSE = True
LEARNING_RATE = 0.001
EPOCHS = 30 # 30
FORCE_RETRAIN = True

TO_TEST = 0

def main():
    # Import benign model
    benign_model = network_definition
    benign_model.load_state_dict(torch.load(benign_model_file_path, map_location=device))

    # Import subject model
    subject_model = network_definition
    subject_model.load_state_dict(torch.load(subject_model_file_path, map_location=device))

    print("\nTesting the " + model_string + " model for backdoors...")

    if TO_TEST == 0 or TO_TEST == 1:
        # Retrain the subject model and test the weights to deteremine if there is a back door
        backdoor = rt.main(network=network_definition, 
                        subject=subject_model, 
                        trainset=trainset, 
                        testset=testset, 
                        retrained=retrained_model_file_path, 
                        n_control_models=N_CONTROL_MODELS, 
                        model_string = model_string,
                        learning_rate=LEARNING_RATE, 
                        epochs=EPOCHS, 
                        threshold=THRESHOLD, 
                        verbose=VERBOSE
                        )
        if backdoor:
            print(f'\nIt is possible that the network has a weight-based backdoor, becuase the percentage of outlier neurons is above the {THRESHOLD} threshold.\n')
        else:
            print('\nIt is unlikely that the network has a weight-based backdoor.\n')
        

    if TO_TEST == 0 or TO_TEST == 2:
        Subject_Model, Benign_Model = subject_model.to(device), benign_model.to(device)
        # Subject_Model, Benign_Model = subject_model.to('cpu'), benign_model.to('cpu')
        # Test robustness of model using robustness.py tests to determine if there is a backdoor
        robustness_test_results = []
        for i in range(13):
            robust = rb.test_robust(benign=Benign_Model, 
                                    subject=Subject_Model, 
                                    dataset=testset, 
                                    test=i, 
                                    num_img=NUM_IMG, 
                                    eps=EPS, 
                                    threshold=THRESHOLD, 
                                    mnist=mnist,
                                    device = device,
                                    # device = 'cpu', 
                                    verbose=VERBOSE)
            robustness_test_results.append(robust)
            print("Robustness test " + str(i) + ": " + str(robust))

        # Grand vote on robustness
        robustness = max(set(robustness_test_results), key=robustness_test_results.count)
        if robustness:
            print(f'\nWe conclude that the network does not have a backdoor.\n')
        else:
            print('\nWe conclude that the network has a backdoor\n')

    if TO_TEST == 0 or TO_TEST == 3:
        # Fine tuning tests - gaussian noise, retraining with dropout, neural attention distillation (which classes have backdoor)
        cbd = bd.backdoor_forget(model=model_string,
                                subject_model=subject_model,
                                subject_model_filename=subject_model_file_path,
                                trainset=trainset,
                                testset=testset,
                                force_retrain=FORCE_RETRAIN
        )
        print(f'\nThese classes likely have a backdoor: {cbd}\n')

    if TO_TEST == 0 or TO_TEST == 4:
        # Regenerate the trigger
        if CIFAR100:
            classes = cbd
        else:
            classes = [i for i in range(10)]

        trigger = tss.func_trigger_synthesis(MODELNAME=subject_model_file_path,
                                            MODELCLASS=model_string,
                                            TRIGGERS=triggers,
                                            CLASSES=classes,
                                            CIFAR100=CIFAR100
        )
    return 0

if __name__ == '__main__':
    main()
