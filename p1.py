import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import glob
import os
import numpy as np
from PIL import Image
import sys
from torchvision import models

loader = transforms.Compose([
transforms.ToTensor()])


class MNIST(Dataset):
    def __init__(self, transform=None, mode = 'train', root = ''):
        """ Intialize the MNIST dataset """
        self.images = None
        self.labels = None
        self.filenames = []
        self.mode = mode
        self.transform = transform
        self.root = root

        filenames = glob.glob(os.path.join(self.root, '*'))
        if mode == 'train':
          for i in range(50):
            for j in range(450):
              image = Image.open('p1_data/train_50/' + str(i) + '_' + str(j)+'.png').convert('RGB')
              image = loader(image).unsqueeze(0)
              img = image.to('cpu', torch.float)
              self.filenames.append((str(i) + '_' + str(j)+'.png', i))
        if mode == 'val':
          for i in range(50):
            for j in range(450, 500):
              image = Image.open('p1_data/val_50/' + str(i) + '_' + str(j)+'.png').convert('RGB')
              image = loader(image).unsqueeze(0)
              img = image.to('cpu', torch.float)
              self.filenames.append((str(i) + '_' + str(j)+'.png', i))
        if mode == 'test':
          for fn in filenames:
            image = Image.open(os.path.join(self.root,fn)).convert('RGB')
            image = loader(image).unsqueeze(0)
            img = image.to('cpu', torch.float)
            tmp = fn
            self.filenames.append((fn, 0))
        self.len = len(self.filenames)
                              
    def __getitem__(self, index):
        """ Get a sample from the dataset """
        image_fn, label = self.filenames[index]
        image = Image.open(image_fn)
            
        if self.transform is not None:
            image = self.transform(image)
        parse = image_fn.split('/')

        return image, label, parse[-1]

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len


# Create the MNIST dataset. 
mean = [0.5, 0.5, 0.5]
std = [0.1, 0.1, 0.1]
# mytransform = T.Compose([  T.ToTensor()])


testtransform = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 128)
    # transforms.Resize((128, 128)),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
# load the testset
testset = MNIST(transform=testtransform, mode = 'test', root = sys.argv[1])

print('# images in testset:', len(testset)) # Should print 10000


testset_loader = DataLoader(testset, batch_size=50, shuffle=False, num_workers=1)


# Use GPU if available, otherwise stick with cpu
use_cuda = torch.cuda.is_available()
torch.manual_seed(123)
device = torch.device("cuda" if use_cuda else "cpu")
print('Device used:', device)


model = models.resnet152(pretrained=True).to(device)

# print(model.fc)
model.fc = nn.Linear(2048, 50)

model.to(device)

def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)
    
def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)


def train(model, epoch, log_interval=100):
    optimizer = optim.SGD(model.parameters(), lr= 1e-3, weight_decay=1e-5, momentum=0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [5,10,20], gamma=0.1, last_epoch=-1)
    criterion = nn.CrossEntropyLoss()
    model.train()  # Important: set training mode
    
    iteration = 0
    for ep in range(epoch):
        for batch_idx, (data, target, _) in enumerate(trainset_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if iteration % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    ep, batch_idx * len(data), len(trainset_loader.dataset),
                    100. * batch_idx / len(trainset_loader), loss.item()))
            iteration += 1
        scheduler.step()
            
        test(model) # Evaluate at the end of each epoch
        save_checkpoint('drive/MyDrive/DLCVHW1/p1/p1-%i.pth' % ep, model, optimizer)


def test(model):
    criterion = nn.CrossEntropyLoss()
    model.eval()  # Important: set evaluation mode
    test_loss = 0
    correct = 0
    answer = []
    with torch.no_grad(): # This will free the GPU memory used for back-prop
        for data, target, name in testset_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            for i in range(len(data)):
              # print(name[i], pred[i][0].item())
              answer.append((name[i], pred[i][0].item()))

    test_loss /= len(testset_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(testset_loader.dataset),
        100. * correct / len(testset_loader.dataset)))
    return answer

"""It's time to train the model!"""

# create a new model
model = model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# load from the final checkpoint
load_checkpoint('p1-14.pth', model, optimizer)

# should give you the final model accuracy
answer = test(model)

import pandas as pd
a = pd.DataFrame(answer, columns=['image_id', 'label'])
a.to_csv(sys.argv[2],index=False)