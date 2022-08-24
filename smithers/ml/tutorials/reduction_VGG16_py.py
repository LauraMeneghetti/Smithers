######################################################
#IMPORTS
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

import warnings
warnings.filterwarnings("ignore")



import datetime


sys.path.insert(0, '../')
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')






######################################################
os.nice(15)
print('', flush=True)
print('', flush=True)
print('', flush=True)
print('', flush=True)
print('', flush=True)
print('', flush=True)
print('', flush=True)
print('#########################################', flush=True)
print('NUOVA ESECUZIONE ALLE ORE '+str(datetime.datetime.now().time()), flush=True)
print('#########################################', flush=True)
print('', flush=True)
print('', flush=True)
print('', flush=True)
print('', flush=True)
print('', flush=True)
print('', flush=True)
print('', flush=True)






######################################################
# DEFINIZIONE VGG
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











 


######################################################
# EXPORT E IMPORT DEL CHECKPOINT
def save_checkpoint_torch(epoch, model, path, optimizer):
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, path)


def load_checkpoint(model, checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])
    











######################################################
# LOADING CIFAR10
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
train_labels = torch.tensor(train_loader.dataset.targets).to(device) #MODIF
targets = list(train_labels)
#print(type(test_dataset)) #MODIF
#print(targets)









#############################################
# TRAINING
""" print('inizio training', flush=True)
print('Training iniziato') #MODIF
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




save_checkpoint_torch(60, VGGnet, '/u/s/szanin/Smithers/smithers/ml/tutorials/check_vgg_cifar10_60_stefano.pth.tar', optimizer)
 """















######################################################
# LOADING CHECKPOINT
#pretrained = '/u/s/szanin/Smithers/smithers/ml/tutorials/check_vgg_cifar10_60.pth.tar'#Laura's
pretrained = '/u/s/szanin/Smithers/smithers/ml/tutorials/check_vgg_cifar10_60_v2.pth.tar' #Stefano's
model = VGGnet
load_checkpoint(model, pretrained)
seq_model = get_seq_model(model)
model = model.to(device) #MODIF
seq_model = seq_model.to(device) #MODIF











######################################################
# REDUCTION OF VGG16
total = 0
correct = 0
count = 0
seq_model.eval()
for test, y_test in iter(test_loader):
#Calculate the class probabilities (softmax) for img
    with torch.no_grad():
        output = seq_model(test.to(device)) #MODIF
        ps = torch.exp(output)
        _, predicted = torch.max(output.data,1)
        total += y_test.size(0)
        correct += (predicted == y_test.to(device)).sum().item() #MODIF
        count += 1
        #print("Accuracy of network on test images is {:.4f}....count: {}".format(100*correct/total,  count ))
        if count%50 == 0:
            print("Accuracy of network on test images is {:.4f}....count: {}".format(100*correct/total,  count ), flush=True)




from smithers.ml.netadapter import NetAdapter

cutoff_idx = 7
red_dim = 50 
red_method = 'POD' 
inout_method = 'FNN'
n_class = 10 #MODIF

netadapter = NetAdapter(cutoff_idx, red_dim, red_method, inout_method)
red_model = netadapter.reduce_net(seq_model, train_dataset, train_labels, train_loader, n_class).to(device) #MODIF
print(red_model, flush=True)













from smithers.ml.utils import Total_param, Total_flops
from smithers.ml.utils import compute_loss, train_kd


rednet_storage = torch.zeros(3)
rednet_flops = torch.zeros(3)

rednet_storage[0], rednet_storage[1], rednet_storage[2] = [
    Total_param(red_model.premodel),
    Total_param(red_model.proj_model),
    Total_param(red_model.inout_map)]

rednet_flops[0], rednet_flops[1], rednet_flops[2] = [
    Total_flops(red_model.premodel, device),
    Total_flops(red_model.proj_model, device),
    Total_flops(red_model.inout_map, device)]

total = 0
correct = 0
count = 0
for test, y_test in iter(test_loader):
#Calculate the class probabilities (softmax) for img
    with torch.no_grad():
        output = red_model(test)
        ps = torch.exp(output)
        _, predicted = torch.max(output.data,1)
        total += y_test.size(0)
        correct += (predicted == y_test.to(device)).sum().item() #MODIF
        count += 1
        #print("Accuracy of network on test images is {:.4f}....count: {}".format(100*correct/total,  count ))
        if count%50 == 0:
            print("Accuracy of network on test images is {:.4f}....count: {}".format(100*correct/total,  count), flush=True)


print(
      'Pre nnz = {:.2f}, proj_model nnz={:.2f}, FNN nnz={:.4f}'.format(
      rednet_storage[0], rednet_storage[1],
      rednet_storage[2]), flush=True)
print(
      'flops:  Pre = {:.2f}, proj_model = {:.2f}, FNN ={:.2f}'.format(
       rednet_flops[0], rednet_flops[1], rednet_flops[2]), flush=True)

optimizer = torch.optim.Adam([{
            'params': red_model.premodel.parameters(),
            'lr': 1e-4
            }, {
            'params': red_model.proj_model.parameters(),
            'lr': 1e-5
            }, {
            'params': red_model.inout_map.parameters(),
            'lr': 1e-5
            }])

train_loss = []
test_loss = []
train_loss.append(compute_loss(red_model, device, train_loader))
test_loss.append(compute_loss(red_model, device, test_loader))

        
epochs = 10
filename = './cifar10_VGG16_RedNet'+\
            '_cutIDx_%d.pth'%(cutoff_idx)

""" if os.path.isfile(filename):
    [rednet_pretrained, train_loss,test_loss] = torch.load(filename)
    red_model.load_state_dict(rednet_pretrained)
    print('rednet trained {} epoches is loaded'.format(epochs), flush=True)
else:
    train_loss = []
    test_loss = []
    train_loss.append(compute_loss(red_model, device, train_loader))
    test_loss.append(compute_loss(red_model, device, test_loader))
    for epoch in range(1, epochs + 1):
        print('EPOCH {}'.format(epoch), flush=True)
        train_loss.append(
                train_kd(red_model,
                model,
                device,
                train_loader,
                optimizer,
                train_max_batch=200,
                alpha=0.1,
                temperature=1.,
                epoch=epoch))
        test_loss.append(compute_loss(red_model, device, test_loader))
    torch.save([red_model.state_dict(), train_loss, test_loss], filename) """

for epoch in range(1, epochs + 1):                       #da qui alla fine era dentro l'else commentato
    print('EPOCH {}'.format(epoch), flush=True)
    train_loss.append(
            train_kd(red_model,
            model,
            device,
            train_loader,
            optimizer,
            train_max_batch=200,
            alpha=0.1,
            temperature=1.,
            epoch=epoch))
    test_loss.append(compute_loss(red_model, device, test_loader))
torch.save([red_model.state_dict(), train_loss, test_loss], filename)








######################################################