from models.definitions import MNISTNet, CIFAR10Net, CIFAR100Net
from torchvision import datasets, transforms

########### PARAMETER FOR USER INPUT ###########
# TEST_CASE Legend: 1 = MNIST, 2 = CIFAR10, 3 = CIFAR100
MODEL_STRING_MAP = {
    1: 'MNIST',
    2: 'CIFAR10',
    3: 'CIFAR100',
}
TEST_CASE = 1 

# Enter the file path where the model to be tested is stored
SUBJECT_MODEL_FILE_PATH = "models/subject/mnist_backdoored_1.pt"
########### PARAMETER FOR USER INPUT ###########


########### EDITABLE PARAMETERS - feel free to change these to your specifications ###########
# Global retraining settings - set to False by default because we already have retrained models (reduce run time)
FORCE_RETRAIN = False   

#the percentage of input data used for trigger synthesis on CIFAR100 models, a float in [0.04 , 1]
#CIFAR100_pct=1: use the full training data and will run > 7horus on GPU !!!
#CIFAR100_pct=0.04: use 4% of training data and will run ~1h20min for trigger synthesis, 3-4 hours in total
#this parameter will not affect MNIST and CIFAR10 cases
CIFAR100_pct = 0.04

# Modelling parameters
EPS = 0.2
THRESHOLD = 0.1
N_CONTROL_MODELS = 2
LEARNING_RATE = 0.001
EPOCHS = 30

# Printing global parameter
VERBOSE = True

########### NON-EDITABLE PARAMETERS ###########
TO_TEST = 0

# Provide filepaths
DATA_FILE_PATH = "data/"

# Location of log files
LOG_DIR = 'logs'
BASE_LOG_FILENAME = 'bd_detection'

NUM_IMG = 10000 #number of images in test set
