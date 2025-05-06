import torch.nn as nn
import torch

#Defining Generator Model
class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.encoder = nn.Sequential(
            #block 1
            nn.Conv2d(3,    64, kernel_size=1, padding=1),
            nn.ReLU(inplace = True),
            nn.Conv2d(64,   64, kernel_size=3, padding=1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2,2),
            #block 2
            nn.Conv2d(64,  128, kernel_size=3, padding=1),
            nn.ReLU(inplace = True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2,2),
            #block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2,2),
            #block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace = True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace = True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2,2),
            #block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace = True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace = True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace = True)
        )
        self.deconv = nn.Sequential(
            #block 6
            nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace = True),
            nn.Upsample(scale_factor=2),
            #block 7
            nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace = True),
            nn.Upsample(scale_factor=2),
            #block 8
            nn.ConvTranspose2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace = True),
            nn.Upsample(scale_factor=2),
            #block 9
            nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace = True),
            nn.Upsample(scale_factor=2),
            #block 10
            nn.ConvTranspose2d(128,  64, kernel_size=3, padding=1),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(64,   64, kernel_size=3, padding=1),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(64,    1, kernel_size=1, padding=0)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x = self.encoder(x)
        x = self.deconv(x)
        x = self.sigmoid(x)
        return x

# Defining Discriminator Model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.conv = nn.Sequential(
            #block 1
            nn.Conv2d(4,   3, kernel_size=1, padding=1), # 4 input channels - 3 original + one original image
            nn.ReLU(inplace=True),
            nn.Conv2d(3,  32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            #block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            #block 3
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*24*32, 100),
            nn.Tanh(),
            nn.Linear(100,2),
            nn.Tanh(),
            nn.Linear(2,1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x
