from models.definitions import MNISTNet, CIFAR10Net, CIFAR100Net

# Global setting on whether to force retraining if the model file already exists
FORCE_RETRAIN = False

# Global setting on printing full outputs
VERBOSE = False

# Location of log files
LOG_DIR = 'logs'
BASE_LOG_FILENAME = 'bd_detection'

THRESHOLD = 0.1
EPOCHS = 30
LR = 0.001

TEST_CASE = 1

# TEST CASES
if TEST_CASE == 1:
    LOGGER = "Running Test Case 1 - MNIST"
    MODEL_STRING = "MNIST"
    NETWORK_DEFINITION = MNISTNet()
    BENIGN_MODEL_FILE_PATH = "models/benign/mnist.pt"
    SUBJECT_MODEL_FILE_PATH = "models/subject/mnist_backdoored_1.pt"
    RETRAINED_MODEL_FILE_PATH = "./models/retrained/retrained_mnist_"
    triggers = "./backdoor_triggers/mnist_backdoored_1/"
    # triggers = triggers
    trainset = datasets.MNIST(data_file_path, train=True, download=True, transform=transforms.ToTensor())
    testset = datasets.MNIST(data_file_path, train=False, download=True, transform=transforms.ToTensor())
    mnist = True
    CIFAR100_pct=1

elif TEST_CASE == 2:
    LOGGER = "Running Test Case 2 - CIFAR10"
    MODEL_STRING = "CIFAR10"
    NETWORK_DEFINITION = CIFAR10Net()
    BENIGN_MODEL_FILE_PATH = "models/benign/benign_CIFAR10.pt"
    SUBJECT_MODEL_FILE_PATH = "models/subject/best_model_CIFAR10_10BD.pt"
    RETRAINED_MODEL_FILE_PATH = "./models/retrained/retrained_CIFAR10_10BD_"
    triggers = "./backdoor_triggers/best_model_CIFAR10_10BD/"
    # triggers = triggers
    trainset = datasets.CIFAR10(data_file_path, train=True, download=True, transform=transforms.ToTensor())
    testset = datasets.CIFAR10(data_file_path, train=False, download=True, transform=transforms.ToTensor())
    mnist = False
    CIFAR100_pct=1

elif TEST_CASE == 3:
    LOGGER = "Running Test Case 3 - CIFAR100"
    MODEL_STRING = "CIFAR100"
    NETWORK_DEFINITION = CIFAR100Net()
    BENIGN_MODEL_FILE_PATH = "models/benign/CIFAR100_seed3.pt"
    SUBJECT_MODEL_FILE_PATH = "models/subject/CIFAR100_bn_BD5.pt"
    RETRAINED_MODEL_FILE_PATH = "./models/retrained/retrained_CIFAR100_"
    triggers = "./backdoor_triggers/CIFAR100_bn_BD5/"
    # triggers = triggers
    trainset = datasets.CIFAR100(data_file_path, train=True, download=True, transform=transforms.ToTensor())
    testset = datasets.CIFAR100(data_file_path, train=False, download=True, transform=transforms.ToTensor())
    mnist = False
    CIFAR100_pct = 0.04  # percentage of input data to use