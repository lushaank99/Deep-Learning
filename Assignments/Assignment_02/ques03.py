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
imageSize = 64
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
    
    elif classname.find('BatchNorm') != -1:
        
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)


class Flatten(nn.Module):
    
    def forward(self, input):
        return input.view(input.size(0), -1)


class Generator(nn.Module):
    
    def __init__(self, feature=False):
        
        super(Generator, self).__init__()
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


class Discriminator(nn.Module):
  
  def __init__(self):

          super(Discriminator, self).__init__()

          self.main = nn.Sequential(
              nn.Conv2d(3, 64, 4, 2, 1, bias = False),
              nn.LeakyReLU(0.2, inplace = True),
              nn.Conv2d(64, 128, 4, 2, 1, bias = False),
              nn.BatchNorm2d(128),
              nn.LeakyReLU(0.2, inplace = True),
              nn.Conv2d(128, 256, 4, 2, 1, bias = False),
              nn.BatchNorm2d(256),
              nn.LeakyReLU(0.2, inplace = True),
              nn.Conv2d(256, 512, 4, 2, 1, bias = False),
              nn.BatchNorm2d(512),
              nn.LeakyReLU(0.2, inplace = True),
              nn.Conv2d(512, 1, 4, 1, 0, bias = False),
              nn.Sigmoid()
          )

  def forward(self, input):
      
        return self.main(input)

Gnetwork = Generator().to(device)
Gnetwork.apply(weights_init)

Dnetwork = Discriminator().to(device)
Dnetwork.apply(weights_init)

criterion = nn.BCELoss()

Doptim = optim.Adam(Dnetwork.parameters(), lr=learning_rate, betas=(0.2, 0.999))
Goptim = optim.Adam(Gnetwork.parameters(), lr=learning_rate, betas=(0.2, 0.999))

Dscheduler = torch.optim.lr_scheduler.MultiStepLR(Doptim, milestones=[5,10,15,20], gamma=0.1)
Gscheduler = torch.optim.lr_scheduler.MultiStepLR(Goptim, milestones=[5,10,15,20], gamma=0.1)


img_list = []
generator_losses = []
discriminator_losses = []

for epoch in range(25):
    
    Dscheduler.step()
    Gscheduler.step()
    
    for i, data in enumerate(dataloader, 0):

        Dnetwork.zero_grad()
        real = data[0].to(device)
        batch_size = real.size(0)
        target = torch.full((batch_size,), 1, device=device)
        
        output = Dnetwork(real).view(-1)
        error_Dreal = criterion(output, target)
        error_Dreal.backward()

        fake = Gnetwork(real)
        target.fill_(0)
        
        output = Dnetwork(fake.detach()).view(-1)
        error_Dfake = criterion(output, target)
        error_Dfake.backward()
        
        error_D = error_Dreal + error_Dfake
        Doptim.step()
        
        Gnetwork.zero_grad()
        target.fill_(1)
        output = Dnetwork(fake).view(-1)
        
        error_G = criterion(output, target)
        error_Auto = criterion(fake, real)
        error_G = error_G + error_Auto
        error_G.backward()
        Goptim.step()
        
        print('Discriminator Loss:\t'+ str(error_D.item()) +'\tGenerator Loss:\t'+ str(error_G.item()))

        generator_losses.append(error_G.item())
        discriminator_losses.append(error_D.item())

        if ((epoch == 25-1) and (i == len(dataloader)-1)):
            
            with torch.no_grad():
                fake = Gnetwork(real).detach().cpu()
            
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

plt.figure(figsize=(10,10))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(generator_losses,label="Generator")
plt.plot(discriminator_losses,label="Discriminator")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig('q3_graph.png')
plt.show()

real_batch = next(iter(dataloader))

plt.figure(figsize=(15,15))

plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.savefig('q3_images.png')
plt.show()