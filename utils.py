import torch
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.autograd import Variable
from torch.utils.serialization import load_lua
from torch.utils.data import DataLoader
from torchvision import transforms, models, datasets
import pdb
import matplotlib.pyplot as plt
import glob
import os
from PIL import Image
import bcolz
import pdb
from bcolz_array_iterator import BcolzArrayIterator


def get_avg_pool_vgg(stop_layer):
    vgg = models.vgg.vgg16(pretrained = True)

    layers = []
    for i, layer in enumerate(vgg.features):
        if isinstance(layer,nn.MaxPool2d):
            #replace MaxPool layers with AvgPool
            layers.append(nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)))
        else:
            layers.append(layer)
        if stop_layer and i == stop_layer:
            pass
            #break
    
    #create a new model with our modified layers
    vgg_mod = nn.Sequential(*layers)
    
    #load the pretrained vgg weights into our model
    vgg_mod.load_state_dict(vgg.features.state_dict())
    for param in vgg_mod.parameters():param.requires_grad = False
    return vgg_mod
    '''
    if stop_layer:
        ls = [l for i,l in enumerate(vgg_mod.children()) if i < stop_layer+1]
        mod = nn.Sequential(*ls)
        for param in mod.parameters():param.requires_grad = False
        return mod
    else:
        for param in vgg_mod.parameters():param.requires_grad = False
        return vgg_mod
    '''

m = [0.485, 0.456, 0.406]; std = [0.229, 0.224, 0.225]
def toTens(imgs):
    img = torch.from_numpy(imgs.transpose((0, 3, 1, 2)))
    # backward compatibility
    return img.float().div(255)

def norm(img):
    img[:,0, :, :] = (img[:,0, :, :] - m[0]) / std[0]
    img[:,1, :, :] = (img[:,1, :, :] - m[1]) / std[1]
    img[:,2, :, :] = (img[:,2, :, :] - m[2]) / std[2]
    return img

def toTensAndNorm(imgs):
    return norm(toTens(imgs))

def deNormTrans(img):
    m = [0.485, 0.456, 0.406]; std = [0.229, 0.224, 0.225]
    img[0, :, :] = img[0, :, :] * std[0] + m[0]
    img[1, :, :] = img[1, :, :] * std[1] + m[1]
    img[2, :, :] = img[2, :, :] * std[2] + m[2]
    return pilTrans(img)

def denormAndPlot(img):
    img = img.clone().squeeze().cpu()
    img = deNormTrans(img)
    plt.imshow(cCrop(img))
    #plt.imshow(cCrop(pilTrans(img)))
    plt.show()

mse = torch.nn.MSELoss()
noRed = torch.nn.MSELoss(size_average=False)

pilTrans = transforms.ToPILImage()
toTensAndNormTrans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(m, std)])
inputTrans = transforms.Compose([transforms.Scale((256,256)),transforms.ToTensor()])
styleImgTrans = transforms.ToTensor()
cCrop = transforms.CenterCrop(256)