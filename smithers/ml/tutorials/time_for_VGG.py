import torch
import numpy as np
import torchvision
from torch import nn
import sys
import os

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pandas as pd
import torch.optim as optim

from smithers.ml.vgg import VGG
from smithers.ml.utils import get_seq_model

import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings("ignore")


torch.cuda.empty_cache()
import datetime
import time

sys.path.insert(0, '../')
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')



batch_size = 8 #this can be changed
data_path = '../datasets/' 
# transform functions: take in input a PIL image and apply this
# transformations
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
train_dataset = datasets.CIFAR10(root=data_path + 'CIFAR10/',
                                 train=True,
                                 download=True,
                                 transform=transform_train)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_dataset = datasets.CIFAR10(root=data_path + 'CIFAR10/',
                                train=False,
                                download=True,
                                transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)
train_labels = torch.tensor(train_loader.dataset.targets).to(device)
targets = list(train_labels)



def save_checkpoint_torch(epoch, model, path, optimizer):
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, path)


def load_checkpoint(model, checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])

start = time.time()

VGGnet = VGG(    cfg=None,
                 classifier='cifar',
                 batch_norm=False,
                 num_classes=10,
                 init_weights=False,
                 pretrain_weights=None)
VGGnet = VGGnet.to(device) #MODIF
VGGnet.make_layers()
VGGnet._initialize_weights()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(VGGnet.parameters(), lr=0.001, momentum=0.9)



print('inizio training', flush=True)
for epoch in range(60):  # loop over the dataset multiple times
    print("Inizia ora l'epoca "+str(epoch), flush=True)
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)


        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = VGGnet(inputs)
        outputs = outputs[1]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        #if i % 2000 == 1999:    # print every 2000 mini-batches
        if i % 200 == 199:    # print every 200 mini-batches #MODIF
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000), flush=True)
            running_loss = 0.0

end = time.time()
print("L'inizializzazione e il training hanno impiegato {:.1f} secondi".format(end-start))
seq_model = get_seq_model(VGGnet)
VGGnet = VGGnet.to(device)
seq_model = seq_model.to(device)

# accuracy
total = 0
correct = 0
count = 0
VGGnet.eval()
for test, y_test in iter(test_loader):
    with torch.no_grad():
        output = VGGnet(test.to(device)).to(device)
        ps = torch.exp(output)
        _, predicted = torch.max(output.data,1)
        total += y_test.size(0)
        correct += (predicted == y_test.to(device)).sum().item() #MODIF
        count += 1
print("L'inizializzazione e il training hanno impiegato {:.1f} secondi".format(end-start))
print('Accuracy of network on test images is {:.4f}'.format(100*correct/total), flush=True)
