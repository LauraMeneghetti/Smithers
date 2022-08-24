import torch
import numpy as np
import torchvision

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pandas as pd

from smithers.ml.vgg import VGG
import os

import warnings
warnings.filterwarnings("ignore")

os.nice(18)

#########################################




def save_checkpoint(epoch, model, optimizer):
    '''
    Save model checkpoint.
    :param scalar epoch: epoch number
    :param list model: list of constructed classes that compose our network
    :param torch.Tensor optimizer: optimizer chosen
    :return: path to the checkpoint file
    :rtype: str
    '''
    state = {'epoch': epoch, 'model': model, 'optimizer': optimizer}
    filename = 'checkpoint_ssd300.pth.tar'
    torch.save(state, filename)
    return filename








#########################################





VGGnet = VGG(    cfg=None,
                 classifier='standard',
                 batch_norm=False,
                 num_classes=1000,
                 init_weights=True,
                 pretrain_weights=None)
VGGnet.make_layers()
VGGnet._initialize_weights()
VGGnet.load_pretrained_layers(None)






#########################################







import sys
#sys.path.insert(0, '/scratch/lmeneghe/Smithers')
sys.path.insert(0, '../')
from smithers.ml.vgg import VGG
from smithers.ml.utils import get_seq_model

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    

model = VGGnet
#model = model['model']
seq_model = get_seq_model(model)




#########################################






#load CIFAR10 dataset for training and testingpretrained = '../checkpoint_vgg16_custom20.pth.tar'
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
train_labels = torch.tensor(train_loader.dataset.targets)
targets = list(train_labels)





#########################################






total = 0
correct = 0
count = 0
seq_model.eval()
for test, y_test in iter(test_loader):
#Calculate the class probabilities (softmax) for img
    with torch.no_grad():
        output = seq_model(test)
        ps = torch.exp(output)
        _, predicted = torch.max(output.data,1)
        total += y_test.size(0)
        correct += (predicted == y_test).sum().item()
        count += 1
        print("Accuracy of network on test images is {:.4f}....count: {}".format(100*correct/total,  count ))






#########################################





from smithers.ml.netadapter import NetAdapter
cutoff_idx = 7 
red_dim = 50 
red_method = 'POD' 
inout_method = 'FNN'
n_class = 4
netadapter = NetAdapter(cutoff_idx, red_dim, red_method, inout_method)
red_model = netadapter.reduce_net(seq_model, train_dataset, train_labels, train_loader, n_class)
print(red_model)






#########################################




import os
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
        correct += (predicted == y_test).sum().item()
        count += 1
        print("Accuracy of network on test images is {:.4f}....count: {}".format(100*correct/total,  count ))

print(
      'Pre nnz = {:.2f}, proj_model nnz={:.2f}, FNN nnz={:.4f}'.format(
      rednet_storage[0], rednet_storage[1],
      rednet_storage[2]))
print(
      'flops:  Pre = {:.2f}, proj_model = {:.2f}, FNN ={:.2f}'.format(
       rednet_flops[0], rednet_flops[1], rednet_flops[2]))

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

if os.path.isfile(filename):
    [rednet_pretrained, train_loss,test_loss] = torch.load(filename)
    red_model.load_state_dict(rednet_pretrained)
    print('rednet trained {} epoches is loaded'.format(epochs))
else:
    train_loss = []
    test_loss = []
    train_loss.append(compute_loss(red_model, device, train_loader))
    test_loss.append(compute_loss(red_model, device, test_loader))
    for epoch in range(1, epochs + 1):
        print('EPOCH {}'.format(epoch))
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








#########################################







save_checkpoint(10, VGGnet, optimizer)




#########################################












#########################################












#########################################












#########################################












#########################################












#########################################












#########################################












#########################################












#########################################












#########################################












#########################################












#########################################