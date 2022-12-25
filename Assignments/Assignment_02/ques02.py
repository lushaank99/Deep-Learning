from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


batchSize = 32
imageSize = 128
learning_rate = 0.001
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
print(device)


def dataloader_function(datadir):
  
    data_transform = transforms.Compose([transforms.Resize((imageSize,imageSize)), transforms.ToTensor()])
    dataset = datasets.ImageFolder(datadir, transform = data_transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle = True)
    
    return dataloader

# Change the below directory from data to new data for checking given video sample

data_dir = 'new_data/'
dataloader = dataloader_function(data_dir)
print(len(dataloader.dataset.classes))


def weights_init(model):
    
    classname = model.__class__.__name__
    
    if classname.find('Conv') != -1:
        model.weight.data.normal_(0.0, 0.02)

class Flatten(nn.Module):
    
    def forward(self, input):
        return input.view(input.size(0), -1)

class ConvAutoencoder(nn.Module):
    
    def __init__(self, feature=False):
        
        super(ConvAutoencoder, self).__init__()
        self.feature = feature
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)  
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.linear = nn.Linear(128*8*8,100)
        self.t_conv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.t_conv4 = nn.ConvTranspose2d(16, 3, 2, stride=2)


    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        
        if self.feature:
          
          x = Flatten(x)
          return self.linear(x)
        
        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))
        x = F.relu(self.t_conv3(x))
        x = F.sigmoid(self.t_conv4(x))
                
        return x

net = ConvAutoencoder().to(device)
net.apply(weights_init)


criterion = nn.BCELoss()

optimizer = optim.Adam(net.parameters(), lr=learning_rate, betas=(0.2, 0.999))

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5,10,15,20], gamma=0.1)

losses = []

for epoch in range(25):
  
    scheduler.step()
  
    for i, data in enumerate(dataloader, 0):

        images = data[0].to(device)
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()
        
        print('Loss:\t'+ str(loss.item()))
        
        losses.append(loss.item())

plt.figure(figsize=(10,10))
plt.title("Loss Training")
plt.plot(losses,label="Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig('q2_graph.png')
plt.show()