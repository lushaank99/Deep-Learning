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


class Generator(nn.Module):
  
  def __init__(self):
    
          super(Generator, self).__init__()

          self.main = nn.Sequential(
              nn.ConvTranspose2d(100, 512, 4, 1, 0, bias = False),
              nn.BatchNorm2d(512),
              nn.ReLU(True),
              nn.ConvTranspose2d(512, 256, 4, 2, 1, bias = False),
              nn.BatchNorm2d(256),
              nn.ReLU(True),
              nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False),
              nn.BatchNorm2d(128),
              nn.ReLU(True),
              nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False),
              nn.BatchNorm2d(64),
              nn.ReLU(True),
              nn.ConvTranspose2d(64, 3, 4, 2, 1, bias = False),
              nn.Tanh()
          )
        
  
  def forward(self, input):
      
      output = self.main(input)
      return output

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

fixed_noise = torch.randn(64, 100, 1, 1, device=device)

Doptim = optim.Adam(Dnetwork.parameters(), lr=learning_rate, betas=(0.2, 0.999))
Goptim = optim.Adam(Gnetwork.parameters(), lr=learning_rate, betas=(0.2, 0.999))

Dscheduler = torch.optim.lr_scheduler.MultiStepLR(Doptim, milestones=[5,10,15,20], gamma=0.1)
Gscheduler = torch.optim.lr_scheduler.MultiStepLR(Goptim, milestones=[5,10,15,20], gamma=0.1)


img_list = []
generator_losses = []
discriminator_losses = []
iterations = 0

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

        noise = torch.randn(batch_size, 100, 1, 1, device=device)
        fake = Gnetwork(noise)
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
        error_G.backward()
        Goptim.step()
        
        print('Discriminator Loss:\t'+ str(error_D.item()) +'\tGenerator Loss:\t'+ str(error_G.item()))

        generator_losses.append(error_G.item())
        discriminator_losses.append(error_D.item())

        if ((epoch == 25-1) and (i == len(dataloader)-1)):
            
            with torch.no_grad():
                fake = Gnetwork(fixed_noise).detach().cpu()
            
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

plt.figure(figsize=(10,10))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(generator_losses,label="Generator")
plt.plot(discriminator_losses,label="Discriminator")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig('q1_graph.png')
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
plt.savefig('q1_images.png')
plt.show()