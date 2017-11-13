import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.autograd import Variable

class ResidualBlock(nn.Module):
    def __init__(self, num):
        super(ResidualBlock, self).__init__()
        self.c1 = nn.Conv2d(num, num, kernel_size=3, stride=1, padding=1)
        self.c2 = nn.Conv2d(num, num, kernel_size=3, stride=1, padding=1)
        self.b1 = nn.BatchNorm2d(num)
        self.b2 = nn.BatchNorm2d(num)

    def forward(self, x):
        h = F.relu(self.b1(self.c1(x)))
        h = self.b2(self.c2(h))
        return h + x


class SuperResolutionNet(nn.Module):
    def __init__(self):
        super(SuperResolutionNet, self).__init__()
        self.ref_padding = torch.nn.ReflectionPad2d(40)
        self.cs1 = nn.Conv2d(3, 32, kernel_size=9, stride=1, padding=4) 
        self.cs2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1) 
        self.cs3 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1) 
        
        self.b1s1 =nn.BatchNorm2d(32)
        self.b1s2 =nn.BatchNorm2d(64)
        self.b1s3 =nn.BatchNorm2d(64)
        
        self.rs1 =  ResidualBlock(64)
        self.rs2 =  ResidualBlock(64)
        self.rs3 =  ResidualBlock(64)
        self.rs4 =  ResidualBlock(64)
        self.rs5 =  ResidualBlock(64)
        
        self.ds1 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.ds2 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.ds3 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.ds4 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)

        self.b2s1 = nn.BatchNorm2d(64)
        self.b2s2 = nn.BatchNorm2d(64)
        self.b2s3 = nn.BatchNorm2d(64)
        self.b2s4 = nn.BatchNorm2d(64)
        self.d3 = nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4)

    def forward(self, h):
        h = F.relu(self.b1s1(self.cs1(h)))
        h = F.relu(self.b1s2(self.cs2(h)))
        h = F.relu(self.b1s3(self.cs3(h)))
        h = self.rs1(h)
        h = self.rs2(h);
        h = self.rs3(h);
        h = self.rs4(h);
        h = self.rs5(h);
        h = F.relu(self.b2s1(self.ds1(h)))
        h = F.relu(self.b2s2(self.ds2(h)))
        h = F.relu(self.b2s3(self.ds3(h)))
        h = F.relu(self.b2s4(self.ds4(h)))      
        h = self.d3(h)
        return (F.tanh(h) + 1.0) / 2