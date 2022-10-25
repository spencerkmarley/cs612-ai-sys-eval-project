import os
import pathlib
import sys
from termios import TIOCPKT_DOSTOP
import torch
import torch.optim as optim

from torch import nn
from torch.utils.data import DataLoader

from torchvision import datasets, transforms
from torchsummary import summary
from main import TEST_CASE

# Add paths
sys.path.append('.')

# Utility functions
import util
from util import get_pytorch_device, open_model, load_model, save_model, train, test
from util import NAD_train, NAD_test

# Import models with perturbations
from models.definitions import MNISTNet, CIFAR10Net, CIFAR100Net
from models.definitions import CIFAR10_Noise_Net, CIFAR10Net_NeuronsOff, CIFAR10Net_AT
from models.definitions import AT

# Device selection - includes Apple Silicon
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

torch.manual_seed(42)
FORCE_RETRAIN = True


# Function definitions
def has_backdoor(subject_model, test_model, test_loader, device, threshold=0.1):
  ''' TODO update these descriptions
  testing_model: a .pt model, the finetuned subject_model
  threshold: percentage [0,1] threshold to flag whether a class could have a backdoor
  test_type: a string, 'Gaussian noised', 'randomly switching off neurons', 'neural attention distilled'
  '''
  percent_diff = prediction_variance(subject_model, test_model, test_loader, device)
  percent_diff = dict(sorted(percent_diff.items(), key=lambda item: abs(item[1]),reverse=True))

  print('Percentage difference in inferences:')
  for k,v in percent_diff.items():
    print('Class {}: {:.2f}%'.format(k,v*100))
  print()
  
  backdoored_classes = [k for k,v in percent_diff.items() if abs(v)>=threshold]
    
  # if len(backdoored_classes):
    # print('\nThe subject model most likely has a backdoor')
    # print('\n----Most likely backdoored classes by descending order----')
    # print(' '.join(str(k) for k in backdoored_classes))
  # else:
    # print('\nThe subject model most likely does not have a backdoor')
    
  return backdoored_classes

def prediction_variance(subject_model, test_model, test_loader, device):
  '''
  Computes the prediction difference between the subject model and the test model.
  Returns a dictionary of the absolute difference in the range of [0,1]
  '''
  
  dist_subject = util.model.get_pred_distribution(subject_model, test_loader, device)
  dist_test = util.model.get_pred_distribution(test_model, test_loader, device)
  
  pred_diff = {x: abs(dist_test[x] - dist_subject[x])/dist_subject[x] if dist_subject[x] else 0 for x in dist_test}
  return pred_diff

def retrain_model(
  base_model_filename,  # Location of the base PyTorch model to be loaded from file
  retrain_arch,  # Model architecture to retrain - found in models/definitions
  train_loader,  # Training data loader
  test_loader,  # Test data loader
  save_filename = None,  # Location to save the retrained model
  device = None,  # Device to use for training (if None, use get_pytorch_device())
  epochs = 30,  # Training epochs
  lr = 0.001,  # Learning rate
  force_retrain = False,  # Force retrain even if model file exists
  ):
  """ Retrains a model from the base with alterations from the retrain_arch """
  
  if device is None:
    device = get_pytorch_device()
    
  if save_filename is None:
    save_filename = create_save_filename(base_model_filename, retrain_arch)
    save_filename = os.path.join('models', 'retrained', save_filename)
  
  if force_retrain or not os.path.exists(save_filename):
    new_model = load_model(retrain_arch, base_model_filename)
    optimizer = optim.Adam(new_model.parameters(), lr=lr)
    best_accuracy = 0
    
    for epoch in range(epochs):
      print('\n------------- Epoch {} -------------\n'.format(epoch+1))
      train(new_model, train_loader, nn.CrossEntropyLoss(), optimizer, device)
      accuracy, _ = test(new_model, test_loader, nn.CrossEntropyLoss(), device)

      if accuracy > best_accuracy:
        save_model(new_model, save_filename)
        best_accuracy = accuracy
        
  new_model = load_model(retrain_arch, save_filename)
  return new_model
 
def create_save_filename(base_model_filename, retrain_arch, suffix = None):
  """ Creates a filename for saving a retrained model """
  filename_stem = pathlib.Path(base_model_filename).stem
  if suffix is None:
    new_filename = filename_stem + '__' + retrain_arch.__name__ + '.pt'
  else:
    new_filename = filename_stem + '__' + suffix + '.pt'
  return new_filename

def backdoor_forget(model, subject_model, trainset, testset):

  device = get_pytorch_device()

  # Train and test arguments
  train_kwargs = {'batch_size': 100, 'shuffle':True}
  test_kwargs = {'batch_size': 1000}
  train_loader = torch.utils.data.DataLoader(trainset, **train_kwargs)
  test_loader = torch.utils.data.DataLoader(testset, **test_kwargs)

  # Test the accuracy of the subject model loaded
  loss_fn = nn.CrossEntropyLoss()
  test(subject_model, test_loader, loss_fn, device)

  # Get the class distrbution from inference of the clean model
  pred_distribution = util.model.get_pred_distribution(subject_model, test_loader, device)
  print(f'Class distribution from inference of clean model: {pred_distribution}\n\n')

  TO_TEST = 1

  if TO_TEST == 1:
    # TEST 1 - Gaussian noise
    print('TEST 1: Adding Noise Layer')
    if model == "MNIST":
      print("No retrain architecture available") # TODO remove this once below completed
      # model_noise = retrain_model(
      # base_model_filename = subject_model_filename,
      # retrain_arch = CIFAR10_Noise_Net, # TODO add retrain architecture
      # train_loader = train_loader,
      # test_loader = test_loader,
      # force_retrain = FORCE_RETRAIN,
      # )
    elif model == "CIFAR10":
      model_noise = retrain_model(
      base_model_filename = subject_model_filename,
      retrain_arch = CIFAR10_Noise_Net,
      train_loader = train_loader,
      test_loader = test_loader,
      force_retrain = FORCE_RETRAIN,
      )
    elif model == "CIFAR100":
      print("No retrain architecture available") # TODO remove this once below completed
      # model_noise = retrain_model(
      # base_model_filename = subject_model_filename,
      # retrain_arch = CIFAR10_Noise_Net, # TODO add retrain architecture
      # train_loader = train_loader,
      # test_loader = test_loader,
      # force_retrain = FORCE_RETRAIN,
      # )
    
    backdoored_classes = has_backdoor(subject_model, model_noise, test_loader, device)
    if len(backdoored_classes):
        print('The subject model likely has a backdoor')
        print(backdoored_classes)
    else:
        print('The subject model does not have a backdoor')

  elif TO_TEST == 2:
    # TEST 2 - Randomly switch off neurons
    print('TEST 2: Turning Neurons Off')
    if model == "MNIST":
      print("No retrain architecture available") # TODO remove this once below completed
      # model_NeuronsOff = retrain_model(
      # base_model_filename = subject_model_filename,
      # retrain_arch = CIFAR10Net_NeuronsOff, # TODO add retrain architecture
      # train_loader = train_loader,
      # test_loader = test_loader,
      # force_retrain = FORCE_RETRAIN,
      # )
    elif model == "CIFAR10":
      model_NeuronsOff = retrain_model(
      base_model_filename = subject_model_filename,
      retrain_arch = CIFAR10Net_NeuronsOff,
      train_loader = train_loader,
      test_loader = test_loader,
      force_retrain = FORCE_RETRAIN,
      )
    elif model == "CIFAR100":
      print("No retrain architecture available") # TODO remove this once below completed
      # model_NeuronsOff = retrain_model(
      # base_model_filename = subject_model_filename,
      # retrain_arch = CIFAR10Net_NeuronsOff, # TODO add retrain architecture
      # train_loader = train_loader,
      # test_loader = test_loader,
      # force_retrain = FORCE_RETRAIN,
      # )
    
    backdoored_classes = has_backdoor(subject_model, model_NeuronsOff, test_loader, device)
    if len(backdoored_classes):
      print('The subject model likely has a backdoor')
      print(backdoored_classes)
    else:
      print('The subject model does not have a backdoor')


  # TEST 3 - Neural Attention Distillation
  print('TEST 3: Neural Attention Distillation')
  if model == "MNIST":
    print("No retrain architecture available") # TODO add retrain architecture
  elif model == "CIFAR10":
    print("To do")
  elif model == "CIFAR100":
    print("No retrain architecture available") # TODO add retrain architecture

  #
  # TEST 
  #
  
  save_filename = create_save_filename(subject_model_filename, None, 'NAD_Teacher')
  save_filename = os.path.join('models', 'retrained', save_filename)

  model_Teacher = retrain_model(
    base_model_filename = subject_model_filename,
    retrain_arch = CIFAR10Net,
    train_loader = train_loader,
    test_loader = test_loader,
    force_retrain = FORCE_RETRAIN,
    save_filename = save_filename
  )

  # Load the Student and Teacher models
  model_student = load_model(CIFAR10Net_AT, subject_model_filename)
  model_Teacher = load_model(CIFAR10Net_AT, save_filename)
  model_student, model_Teacher = model_student.to(device), model_Teacher.to(device)


  # Train the student model 
  model_Teacher.eval()

  for param in model_Teacher.parameters():
    param.requires_grad = False

  criterionCl = nn.CrossEntropyLoss()
  criterionAT = AT(p=2)

  save_filename = create_save_filename(subject_model_filename, None, 'NAD_Student')
  save_filename = os.path.join('models', 'retrained', save_filename)

  FORCE_RETRAIN = False
  if FORCE_RETRAIN or not os.path.exists(save_filename):
    optimizer = optim.Adam(model_student.parameters(), lr = 0.001)
    epochs = 30
    best_accuracy = 0
    for epoch in range(epochs):
      print('\n------------- Epoch {} of student model training-------------\n'.format(epoch+1))
      NAD_train(model_student, model_Teacher, optimizer, criterionCl, criterionAT, train_loader)
      accuracy = NAD_test(model_student, test_loader)

      if accuracy > best_accuracy:
        save_model(model_student, save_filename)

  model_student = load_model(CIFAR10Net, save_filename)

  backdoored_classes = has_backdoor(subject_model, model_student, test_loader, device)
  if len(backdoored_classes):
      print('The subject model likely has a backdoor')
      print(backdoored_classes)
  else:
      print('The subject model does not have a backdoor')

if __name__ == '__main__':
    data_file_path = "./data/"
    
    TEST_CASE = 1

    if TEST_CASE == 1:
      # Load the subject model
      subject_model_filename = "./models/subject/mnist_backdoored_1.pt"
      subject_model = MNISTNet()
      subject_model.load_state_dict(torch.load(subject_model_filename, map_location=device))
      trainset = datasets.MNIST(data_file_path, train=True, download=True, transform=transforms.ToTensor())
      testset = datasets.MNIST(data_file_path, train=False, download=True, transform=transforms.ToTensor())
      model = "MNIST"
    
    elif TEST_CASE == 2:
      # Load the subject model
      subject_model_filename = "./models/subject/best_model_CIFAR10_10BD.pt"
      subject_model = CIFAR10Net()
      subject_model.load_state_dict(torch.load(subject_model_filename, map_location=device))
      trainset = datasets.CIFAR10(data_file_path, train=True, download=True, transform=transforms.ToTensor())
      testset = datasets.CIFAR10(data_file_path, train=False, download=True, transform=transforms.ToTensor())
      model = "CIFAR10"
    
    elif TEST_CASE == 3:
      # Load the subject model
      subject_model_filename = "./models/subject/CIFAR100_bn_BD5.pt"
      subject_model = CIFAR100Net()
      subject_model.load_state_dict(torch.load(subject_model_filename, map_location=device))
      trainset = datasets.CIFAR100(data_file_path, train=True, download=True, transform=transforms.ToTensor())
      testset = datasets.CIFAR100(data_file_path, train=False, download=True, transform=transforms.ToTensor())
      model = "CIFAR100"

    backdoor_forget(model, subject_model, trainset, testset)
