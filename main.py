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

# Provide filepaths
data_file_path = "data/"

# triggers = c.triggers
# if not os.path.exists(triggers):
#     os.makedirs(triggers)

# Provide test cases

logger.info(c.LOGGER)
MODEL_STRING = c.MODEL_STRING
NETWORK_DEFINITION = c.NETWORK_DEFINITION
BENIGN_MODEL_FILE_PATH = c.BENIGN_MODEL_FILE_PATH
SUBJECT_MODEL_FILE_PATH = c.SUBJECT_MODEL_FILE_PATH
RETRAINED_MODEL_FILE_PATH = c.RETRAINED_MODEL_FILE_PATH
triggers = c.
trainset = c.
testset = c.
mnist = c.
CIFAR100_pct = c.

# Set parameters
NUM_IMG = 10000 #number of images in test set
EPS = 0.2
THRESHOLD = 0.1
N_CONTROL_MODELS = 2
VERBOSE = True
LEARNING_RATE = 0.001
EPOCHS = 30 # 30
FORCE_RETRAIN = c.FORCE_RETRAIN

TO_TEST = 0

def main():
    # Import benign model
    benign_model = NETWORK_DEFINITION
    benign_model.load_state_dict(torch.load(BENIGN_MODEL_FILE_PATH, map_location=device))

    # Import subject model
    subject_model = NETWORK_DEFINITION
    subject_model.load_state_dict(torch.load(SUBJECT_MODEL_FILE_PATH, map_location=device))

    logger.info("Testing the " + MODEL_STRING + " model for backdoors...")

    if TO_TEST == 0 or TO_TEST == 1:
        # Retrain the subject model and test the weights to deteremine if there is a back door
        backdoor = rt.main(network=NETWORK_DEFINITION, 
                        subject=subject_model, 
                        trainset=trainset, 
                        testset=testset, 
                        retrained=RETRAINED_MODEL_FILE_PATH, 
                        n_control_models=N_CONTROL_MODELS, 
                        model_string = MODEL_STRING,
                        learning_rate=LEARNING_RATE, 
                        epochs=EPOCHS, 
                        threshold=THRESHOLD, 
                        verbose=VERBOSE
                        )
        if backdoor:
            logger.info(f'It is possible that the network has a weight-based backdoor, becuase the percentage of outlier neurons is above the {THRESHOLD} threshold.\n')
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
            logger.info("Robustness test " + str(i) + ": " + str(robust))

        # Grand vote on robustness
        robustness = max(set(robustness_test_results), key=robustness_test_results.count)
        if robustness:
            logger.info(f'\nWe conclude that the network does not have a backdoor.\n')
        else:
            logger.info('\nWe conclude that the network does have a backdoor\n')

    if TO_TEST == 0 or TO_TEST == 3:
        # Fine tuning tests - gaussian noise, retraining with dropout, neural attention distillation (which classes have backdoor)
        cbd = bd.backdoor_forget(model=MODEL_STRING,
                                subject_model=subject_model,
                                subject_model_filename=SUBJECT_MODEL_FILE_PATH,
                                trainset=trainset,
                                testset=testset,
                                force_retrain=FORCE_RETRAIN
        )
        logger.info(f'\nThese class(es) likely have a backdoor: {cbd}\n')

    if TO_TEST == 0 or TO_TEST == 4:
        #Regenerate the trigger
        logger.info("\nStart trigger synthesis")
        outliers,differs = tss.func_trigger_synthesis(MODELNAME=SUBJECT_MODEL_FILE_PATH,
                                            MODELCLASS=MODEL_STRING,
                                            TRIGGERS=triggers,
                                            CIFAR100_PCT=CIFAR100_pct
        )
        

        
        for fl in sorted(os.listdir(triggers)):
            if fl[-2:]=='pt':
                to_plot = torch.load(os.path.join(triggers,fl),map_location=torch.device('cpu')).detach()
                fig = plt.figure();
                if fl[-5:-3] == 'iv':
                    plt.title(f'Invisible backdoor for: {MODEL_STRING} {fl[:-6]}')
                else:
                    plt.title(f'BadNet backdoor for: {MODEL_STRING} {fl[:-6]}')

                if MODEL_STRING == 'MNIST':
                    plt.imshow(to_plot.permute(1,2,0), cmap = 'gray');
                else:
                    plt.imshow(to_plot.permute(1,2,0));
                plt.savefig(f'{triggers}/{MODEL_STRING}_{fl[:-6]}_{datetime.now().strftime("%Y%m%d-%H%M%S")}.png')
                logger.info(f'\nBackdoor plot for {MODEL_STRING} {fl[:-6]} saved!\n')
                plt.show()
             
    return 0

if __name__ == '__main__':
    main()
