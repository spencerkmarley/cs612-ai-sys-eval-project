from models.definitions import MNISTNet, CIFAR10Net, CIFAR100Net
from torchvision import datasets, transforms

# Global settings
FORCE_RETRAIN = False #  Force retraining if the model file already exists
NUM_IMG = 10000 #number of images in test set
EPS = 0.2
THRESHOLD = 0.1
N_CONTROL_MODELS = 2
VERBOSE = True
LEARNING_RATE = 0.001
EPOCHS = 30 # 30


# Global setting on printing full outputs
VERBOSE = False

# Location of log files
LOG_DIR = 'logs'
BASE_LOG_FILENAME = 'bd_detection'

# Provide filepaths
DATA_FILE_PATH = "data/"


LR = 0.001

TEST_CASE = 1
TO_TEST = 0

# TEST CASES
if TEST_CASE == 1:
    LOGGER = "Running Test Case 1 - MNIST"
    MODEL_STRING = "MNIST"
    NETWORK_DEFINITION = MNISTNet()
    BENIGN_MODEL_FILE_PATH = "models/benign/mnist.pt"
    SUBJECT_MODEL_FILE_PATH = "models/subject/mnist_backdoored_1.pt"
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
    SUBJECT_MODEL_FILE_PATH = "models/subject/best_model_CIFAR10_10BD.pt"
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
    SUBJECT_MODEL_FILE_PATH = "models/subject/CIFAR100_bn_BD5.pt"
    RETRAINED_MODEL_FILE_PATH = "./models/retrained/retrained_CIFAR100_"
    TRIGGERS = "./backdoor_triggers/CIFAR100_bn_BD5/"
    TRAINSET = datasets.CIFAR100(DATA_FILE_PATH, train=True, download=True, transform=transforms.ToTensor())
    TESTSET = datasets.CIFAR100(DATA_FILE_PATH, train=False, download=True, transform=transforms.ToTensor())
    MNIST = False
    CIFAR100_PCT = 0.04  # percentage of input data to use