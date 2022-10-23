import torch
from torch import nn
from .pytorch_functions import get_pytorch_device

def NAD_train(student, teacher, optimizer, criterionCl, criterionAT, dataloader, device=None):
  '''
  Uses the teacher model and the train_loader to train the student model.
  '''
  if device is None:
      device = get_pytorch_device()
  
  size = len(dataloader.dataset)
  student.train()
  
  for batch, (x,y) in enumerate(dataloader):
    x,y = x.to(device), y.to(device)
    
    activation1_s, activation2_s, output_s = student(x)
    cls_loss = criterionCl(output_s, y)

    activation1_t, activation2_t, _ = teacher(x)
    #We weigh the loss of the feature vector to be mugh higher.
    at2_loss = criterionAT(activation2_s, activation2_t.detach()) 
    # at1_loss = criterionAT(activation1_s, activation1_t.detach()) * 0.2
    at_loss = cls_loss + at2_loss

    optimizer.zero_grad()
    at_loss.backward()
    optimizer.step()

    if batch % 100 == 0:
      at_loss, current = at_loss.item(), batch * len(x)
      print('loss: {:.4f} [{}/{}]'.format(at_loss, current, size))

    
def NAD_test(student, dataloader, loss_fn = nn.CrossEntropyLoss(), device=None):
  '''
  Evaluates the student model after it has been trained by the teacher model
  '''
  if device is None:
      device = get_pytorch_device()
      
  size = len(dataloader.dataset)
  num_batches = len(dataloader)
  student.eval()
  loss, correct = 0.0, 0
    
  with torch.no_grad():
    for batch, (x, y) in enumerate(dataloader):
      x, y = x.to(device), y.to(device)
      _, _, pred = student(x)
      loss += loss_fn(pred, y).item()

      result = pred.argmax(1)
      correct += (result == y).type(torch.int).sum().item()

  loss /= num_batches
  correct /= size
  accuracy = 100*correct
  print('Test Result: Accuracy @ {:.2f}%, Avg loss @ {:.4f}\n'.format(accuracy, loss))
  return accuracy