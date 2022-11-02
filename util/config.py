from models.definitions import MNISTNet, CIFAR10Net, CIFAR100Net
from torchvision import datasets, transforms

########### PARAMETER FOR USER INPUT ###########
#TEST_CASE Legend: 1 = MNIST, 2 = CIFAR10, 3 = CIFAR100
TEST_CASE = 1 
SUBJECT_MODEL_FILE_PATH= "models/subject/mnist_backdoored_1.pt" #filepath of uploaded model

########### EDITABLE PARAMETERS - feel free to change these to your specifications ###########
# Global retraining settings - set to False by default because we already have retrained models (reduce run time)
FORCE_RETRAIN = False   

#the percentage of input data used for trigger synthesis on CIFAR100 models, a float in [0.04 , 1]
#CIFAR100_pct=1: use the full training data and will run > 7horus on GPU !!!
#CIFAR100_pct=0.04: use 4% of training data and will run ~1h20min for trigger synthesis, 3-4 hours in total
CIFAR100_pct=0.04

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

# TEST CASES
if TEST_CASE == 1:
    LOGGER = "Running Test Case 1 - MNIST"
    MODEL_STRING = "MNIST"
    NETWORK_DEFINITION = MNISTNet()
    BENIGN_MODEL_FILE_PATH = "models/benign/mnist.pt"
#     SUBJECT_MODEL_FILE_PATH = "models/subject/mnist_backdoored_1.pt"
    RETRAINED_MODEL_FILE_PATH = "./models/retrained/retrained_mnist_"
    TRIGGERS = "./backdoor_triggers/mnist_backdoored_1/"
    TRAINSET = datasets.MNIST(DATA_FILE_PATH, train=True, download=True, transform=transforms.ToTensor())
    TESTSET = datasets.MNIST(DATA_FILE_PATH, train=False, download=True, transform=transforms.ToTensor())
    MNIST = True
    CIFAR100_PCT=1

elif TEST_CASE == 2:
    LOGGER = "Running Test Case 2 - CIFAR10"
    MODEL_STRING = "CIFAR10"
    NETWORK_DEFINITION = CIFAR10Net()
    BENIGN_MODEL_FILE_PATH = "models/benign/benign_CIFAR10.pt"
#     SUBJECT_MODEL_FILE_PATH = "models/subject/best_model_CIFAR10_10BD.pt"
    RETRAINED_MODEL_FILE_PATH = "./models/retrained/retrained_CIFAR10_10BD_"
    TRIGGERS = "./backdoor_triggers/best_model_CIFAR10_10BD/"
    TRAINSET = datasets.CIFAR10(DATA_FILE_PATH, train=True, download=True, transform=transforms.ToTensor())
    TESTSET = datasets.CIFAR10(DATA_FILE_PATH, train=False, download=True, transform=transforms.ToTensor())
    MNIST = False
    CIFAR100_PCT=1

elif TEST_CASE == 3:
    LOGGER = "Running Test Case 3 - CIFAR100"
    MODEL_STRING = "CIFAR100"
    NETWORK_DEFINITION = CIFAR100Net()
    BENIGN_MODEL_FILE_PATH = "models/benign/CIFAR100_seed3.pt"
#     SUBJECT_MODEL_FILE_PATH = "models/subject/CIFAR100_bn_BD5.pt"
    RETRAINED_MODEL_FILE_PATH = "./models/retrained/retrained_CIFAR100_"
    TRIGGERS = "./backdoor_triggers/CIFAR100_bn_BD5/"
    TRAINSET = datasets.CIFAR100(DATA_FILE_PATH, train=True, download=True, transform=transforms.ToTensor())
    TESTSET = datasets.CIFAR100(DATA_FILE_PATH, train=False, download=True, transform=transforms.ToTensor())
    MNIST = False
    CIFAR100_PCT = CIFAR100_pct # percentage of input data to use
