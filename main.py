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

# Utility functions
from util import config as c
from util import get_pytorch_device
from util import logger
from datetime import datetime

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

device = get_pytorch_device()

def main():
    # Load the necessary configuration parameters from the config.py file
    DATA_FILE_PATH = c.DATA_FILE_PATH
    EPOCHS = c.EPOCHS
    EPS = c.EPS
    FORCE_RETRAIN = c.FORCE_RETRAIN
    LEARNING_RATE = c.LEARNING_RATE
    NUM_IMG = c.NUM_IMG
    N_CONTROL_MODELS = c.N_CONTROL_MODELS
    THRESHOLD = c.THRESHOLD
    VERBOSE = c.VERBOSE
    
    # Load which test case we are running
    # This can be found in config.py
    TEST_CASE = c.TEST_CASE
    SUBJECT_MODEL_FILE_PATH = c.SUBJECT_MODEL_FILE_PATH
    MODEL_STRING = c.MODEL_STRING_MAP[TEST_CASE]
    logger.info("Testing the " + MODEL_STRING + " model for backdoors...")
    
    # Initialize parameters based on the test case
    if TEST_CASE == 1: # MNIST
        NETWORK_DEFINITION = MNISTNet()
        BENIGN_MODEL_FILE_PATH = "models/benign/mnist.pt"
    #     SUBJECT_MODEL_FILE_PATH = "models/subject/mnist_backdoored_1.pt"
        RETRAINED_MODEL_FILE_PATH = "./models/retrained/retrained_mnist_"
        TRIGGERS = "./backdoor_triggers/mnist_backdoored_1/"
        TRAINSET = datasets.MNIST(DATA_FILE_PATH, train=True, download=True, transform=transforms.ToTensor())
        TESTSET = datasets.MNIST(DATA_FILE_PATH, train=False, download=True, transform=transforms.ToTensor())
        MNIST = True
        CIFAR100_PCT = 1

    elif TEST_CASE == 2: # CIFAR10
        NETWORK_DEFINITION = CIFAR10Net()
        BENIGN_MODEL_FILE_PATH = "models/benign/benign_CIFAR10.pt"
    #     SUBJECT_MODEL_FILE_PATH = "models/subject/best_model_CIFAR10_10BD.pt"
        RETRAINED_MODEL_FILE_PATH = "./models/retrained/retrained_CIFAR10_10BD_"
        TRIGGERS = "./backdoor_triggers/best_model_CIFAR10_10BD/"
        TRAINSET = datasets.CIFAR10(DATA_FILE_PATH, train=True, download=True, transform=transforms.ToTensor())
        TESTSET = datasets.CIFAR10(DATA_FILE_PATH, train=False, download=True, transform=transforms.ToTensor())
        MNIST = False
        CIFAR100_PCT = 1

    elif TEST_CASE == 3: # CIFAR100
        NETWORK_DEFINITION = CIFAR100Net()
        BENIGN_MODEL_FILE_PATH = "models/benign/CIFAR100_seed3.pt"
    #     SUBJECT_MODEL_FILE_PATH = "models/subject/CIFAR100_bn_BD5.pt"
        RETRAINED_MODEL_FILE_PATH = "./models/retrained/retrained_CIFAR100_"
        TRIGGERS = "./backdoor_triggers/CIFAR100_bn_BD5/"
        TRAINSET = datasets.CIFAR100(DATA_FILE_PATH, train=True, download=True, transform=transforms.ToTensor())
        TESTSET = datasets.CIFAR100(DATA_FILE_PATH, train=False, download=True, transform=transforms.ToTensor())
        MNIST = False
        CIFAR100_PCT = CIFAR100_pct # percentage of input data to use
        
    
    # Create triggers folder if it does not exist
    if not os.path.exists(TRIGGERS):
        os.makedirs(TRIGGERS)

    # Import benign model
    benign_model = NETWORK_DEFINITION
    benign_model.load_state_dict(torch.load(BENIGN_MODEL_FILE_PATH, map_location=device))

    # Import subject model
    subject_model = NETWORK_DEFINITION
    subject_model.load_state_dict(torch.load(SUBJECT_MODEL_FILE_PATH, map_location=device))

    #
    # Load in which test to run
    # 0 - Run all tests
    # 1 - Detection by retraining the weights of the model
    # 2 - Robustness tests by perturbing the model
    # 3 - Forget backdoor by adding noise, retraining with dropout, and Neural Attention Distillation
    # 4 - Trigger synthesis
    #
    # ----------------------------------
    # -            WARNING             -
    # ----------------------------------
    # The tests above can run for a long time especially if on a CPU.  The script currently supports
    # CUDA and MPS (for Apple Silicon) if available.  However, even on GPU, Test 4 will take a very long
    # time on large training sets such as CIFAR100.
    #
    # Where possible, retraining can be avoided if the model has already been retrained and saved.  In this case,
    # the global parameter FORCE_RETRAIN found in config.py can be set to False to avoid retraining.
    # 
    TO_TEST = c.TO_TEST

    if TO_TEST == 0 or TO_TEST == 1:
        # Retrain the subject model and test the weights to deteremine if there is a back door
        backdoor = rt.main(network=NETWORK_DEFINITION, 
                        subject=subject_model, 
                        trainset=TRAINSET, 
                        testset=TESTSET, 
                        retrained=RETRAINED_MODEL_FILE_PATH, 
                        n_control_models=N_CONTROL_MODELS, 
                        model_string = MODEL_STRING,
                        learning_rate=LEARNING_RATE, 
                        epochs=EPOCHS, 
                        threshold=THRESHOLD, 
                        verbose=VERBOSE
                        )
        if backdoor:
            logger.info(f'It is possible that the network has a weight-based backdoor, because the percentage of outlier neurons is above the {THRESHOLD} threshold.\n')
        else:
            logger.info('It is unlikely that the network has a weight-based backdoor.\n')
        

    if TO_TEST == 0 or TO_TEST == 2:
        Subject_Model, Benign_Model = subject_model.to(device), benign_model.to(device)
        # Subject_Model, Benign_Model = subject_model.to('cpu'), benign_model.to('cpu')
        # Test robustness of model using robustness.py tests to determine if there is a backdoor
        robustness_test_results = []
        for i in range(13):
            robust = rb.test_robust(benign=Benign_Model, 
                                    subject=Subject_Model, 
                                    dataset=TESTSET, 
                                    test=i, 
                                    num_img=NUM_IMG, 
                                    eps=EPS, 
                                    threshold=THRESHOLD, 
                                    mnist=MNIST,
                                    device = device,
                                    # device = 'cpu', 
                                    verbose=VERBOSE)
            robustness_test_results.append(robust)
            logger.info("Robustness test " + str(i) + ": " + str(robust))

        # Grand vote on robustness
        robustness = max(set(robustness_test_results), key=robustness_test_results.count)
        if robustness:
            logger.info(f'\nWe conclude that the network does not have a backdoor.\n')
        else:
            logger.info('\nWe conclude that the network does have a backdoor\n')

    if TO_TEST == 0 or TO_TEST == 3:
        # Fine tuning tests - gaussian noise, retraining with dropout, neural attention distillation (which classes have backdoor)
        cbd = bd.backdoor_forget(
                model=MODEL_STRING,
                subject_model=subject_model,
                subject_model_filename=SUBJECT_MODEL_FILE_PATH,
                trainset=TRAINSET,
                testset=TESTSET,
                force_retrain=FORCE_RETRAIN
        )
        logger.info(f'\nThese class(es) likely have a backdoor: {cbd}\n')

    if TO_TEST == 0 or TO_TEST == 4:
        #Regenerate the trigger
        logger.info("\nStart trigger synthesis")
        outliers,differs = tss.func_trigger_synthesis(MODELNAME=SUBJECT_MODEL_FILE_PATH,
                                            MODELCLASS=MODEL_STRING,
                                            TRIGGERS=TRIGGERS,
                                            CIFAR100_PCT=CIFAR100_PCT
        )
        
        for fl in sorted(os.listdir(TRIGGERS)):
            if fl[-2:]=='pt':
                to_plot = torch.load(os.path.join(TRIGGERS,fl),map_location=torch.device('cpu')).detach()
                fig = plt.figure();
                if fl[-5:-3] == 'iv':
                    plt.title(f'Invisible backdoor for: {MODEL_STRING} {fl[:-6]}')
                else:
                    plt.title(f'BadNet backdoor for: {MODEL_STRING} {fl[:-6]}')

                if MODEL_STRING == 'MNIST':
                    plt.imshow(to_plot.reshape(28,28), cmap = 'gray');
                else:
                    plt.imshow(to_plot.permute(1,2,0));
                plt.savefig(f'{TRIGGERS}/{MODEL_STRING}_{fl[:-6]}_{datetime.now().strftime("%Y%m%d-%H%M%S")}.png')
                logger.info(f'\nBackdoor plot for {MODEL_STRING} {fl[:-6]} saved!\n')
                plt.show()
             
    return 0

if __name__ == '__main__':
    main()
