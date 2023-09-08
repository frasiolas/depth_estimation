import torch
import torch.nn as nn
from einops import rearrange
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import glob
import os
from turtle import forward
import torch
from PIL import Image
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import math
import random
from torch.utils.data import Dataset, DataLoader
from dataloader import  NyuDepth_train, NyuDepth_test
from pytorch_msssim import ms_ssim, ssim
import numpy as np
import time
from timm import models as tmod
from loss_function import gradient_loss, SilogLoss


class Decoder(nn.Module):
  def __init__(self):
    kernel_size = 3
    super().__init__()
    self.conv3_19 = nn.Conv2d(384, 384, kernel_size=7, padding=9, groups=384, dilation=3, padding_mode='reflect')
    self.conv3_13 = nn.Conv2d(384, 384, kernel_size=5, padding=6, groups=384, dilation=3, padding_mode='reflect')
    self.conv3_7 = nn.Conv2d(384, 384, kernel_size=3, padding=3, groups=384, dilation=3, padding_mode='reflect')
    self.conv2_3_19 = nn.Conv2d(260, 260, kernel_size=7, padding=9, groups=260, dilation=3, padding_mode='reflect')
    self.conv2_3_13 = nn.Conv2d(260, 260, kernel_size=5, padding=6, groups=260, dilation=3, padding_mode='reflect')
    self.conv2_3_7 = nn.Conv2d(260, 260, kernel_size=3, padding=3, groups=260, dilation=3, padding_mode='reflect')
    self.conv3_3_19 = nn.Conv2d(130, 130, kernel_size=7, padding=9, groups=130, dilation=3, padding_mode='reflect')
    self.conv3_3_13 = nn.Conv2d(130, 130, kernel_size=5, padding=6, groups=130, dilation=3, padding_mode='reflect')
    self.conv3_3_7 = nn.Conv2d(130, 130, kernel_size=3, padding=3, groups=130, dilation=3,  padding_mode='reflect')
    self.conv4_3_19 = nn.Conv2d(90, 90, kernel_size=7, padding=9, groups=90, dilation=3, padding_mode='reflect')
    self.conv4_3_13 = nn.Conv2d(90, 90, kernel_size=5, padding=6, groups=90, dilation=3, padding_mode='reflect')
    self.conv4_3_7 = nn.Conv2d(90, 90, kernel_size=3, padding=3, groups=90, dilation=3,  padding_mode='reflect')
    self.conv5_3_19 = nn.Conv2d(64, 64, kernel_size=7, padding=9, groups=64, dilation=3, padding_mode='reflect')
    self.conv5_3_13 = nn.Conv2d(64, 64, kernel_size=5, padding=6, groups=64, dilation=3, padding_mode='reflect')
    self.conv5_3_7 = nn.Conv2d(64, 64, kernel_size=3, padding=3, groups=64, dilation=3,  padding_mode='reflect')
    self.conv1 = nn.Sequential(
    nn.Conv2d(384, 384, kernel_size, stride=1, padding=1, bias=False),
    nn.BatchNorm2d(384),
    nn.ReLU(inplace=True),
)
    self.conv2 = nn.Sequential(
    nn.Conv2d(464, 260, kernel_size, stride=1, padding=1, bias=False),
    nn.BatchNorm2d(260),
    nn.ReLU(inplace=True),
)
    self.conv3 = nn.Sequential(
    nn.Conv2d(324, 130, kernel_size, stride=1, padding=1, bias=False),
    nn.BatchNorm2d(130),
    nn.ReLU(inplace=True),
)
    self.conv4 = nn.Sequential(
    nn.Conv2d(178, 90, kernel_size, stride=1, padding=1, bias=False),
    nn.BatchNorm2d(90),
    nn.ReLU(inplace=True),
)
    self.conv5 = nn.Sequential(
    nn.Conv2d(122, 64, kernel_size, stride=1, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(inplace=True),
)
    self.conv6 = nn.Sequential(
    nn.Conv2d(64, 1, 1, 1, 0, bias=False),
    nn.Sigmoid()
)

   


  def forward(self,x,dec1,dec2,dec3,dec4):
   
    x = self.conv1(x)
    x= F.interpolate(x,scale_factor=2, mode= 'nearest',)
    x1 = self.conv3_19(x)
    x2 = self.conv3_13(x)
    x3 = self.conv3_7(x)
    x = x1 + x2 + x3 + x
    #---------------
    x = torch.cat((x,dec4),1)
    x = self.conv2(x)
    x= F.interpolate(x,scale_factor=2, mode= 'nearest')
    x1 = self.conv2_3_19(x)
    x2 = self.conv2_3_13(x)
    x3 = self.conv2_3_7(x)
    x = x1 + x2 + x3 + x
     #---------------
    x = torch.cat((x,dec3),1)
    x = self.conv3(x)
    x= F.interpolate(x,scale_factor=2, mode= 'nearest') 
    x1 = self.conv3_3_19(x)
    x2 = self.conv3_3_13(x)
    x3 = self.conv3_3_7(x)
    x = x1 + x2 + x3 + x
     #---------------
    x = torch.cat((x,dec2),1)
    x = self.conv4(x)
    x= F.interpolate(x,scale_factor=2, mode= 'nearest') 
    x1 = self.conv4_3_19(x)
    x2 = self.conv4_3_13(x)
    x3 = self.conv4_3_7(x)
    x = x1 + x2 + x3 + x
    #---------------
    x = torch.cat((x,dec1),1)
    x = self.conv5(x) 
    x= F.interpolate(x,scale_factor=2, mode= 'nearest') 
    x1 = self.conv5_3_19(x)
    x2 = self.conv5_3_13(x)
    x3 = self.conv5_3_7(x)
    x = x1 + x2 + x3 + x

     #--------------
    x= self.conv6(x) * 10
    
    return x



    
features = {}
def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook

class EncoderDecoder(nn.Module):

    def __init__(self,batch_size):
        super().__init__()
        
        self.mvit = tmod.create_model('mobilevit_xs', pretrained=True, num_classes=0, global_pool='')   
        self.Decoder = Decoder()
        stage0 = self.mvit.stages[0].register_forward_hook(get_features('stage_0'))
        stage1 = self.mvit.stages[1].register_forward_hook(get_features('stage_1'))
        stage3 = self.mvit.stages[2].register_forward_hook(get_features('stage_2'))
        stage4 = self.mvit.stages[3].register_forward_hook(get_features('stage_3'))
        self.batch_size = batch_size
    

    def forward(self,x):
        out = self.mvit(x)
        dec1 =features['stage_0']
        dec2 =features['stage_1']
        dec3 =features['stage_2']
        dec4 =features['stage_3']
        out = self.Decoder.forward(out,dec1,dec2,dec3,dec4)
        return out


def main() :
    

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    transformss = transforms.ToPILImage()
    model = EncoderDecoder(batch_size=4)
    num_epochs = 20
    BatchSize = 4
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)
    PATH_SAVE = r'C:\Users\ppapadop\Desktop\depth_vir\mobilenet_dilation.pth'
    #model.load_state_dict(torch.load(PATH_SAVE))
    criterion = nn.L1Loss()
    log_loss = SilogLoss()
   
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    train_dataset = NyuDepth_train()
    train_load = DataLoader(dataset=train_dataset, batch_size=BatchSize, shuffle=True)
    model.cuda()
    model.train() 
    for epoch in range(num_epochs):  # loop over the dataset multiple times   
        t0 = time.time()
        running_loss = 0.0
        for i, data in enumerate(train_load, 0):
            optimizer.zero_grad()
            inputs = data['image'].to(device)
            outputs = data['depth'].to(device)
        
            out = model.forward(inputs)
            out = out.to(device)
            grad_loss = gradient_loss(out/10, outputs/10)
            loss_l1 =criterion(out/10, outputs/10)
            loss_ssim = 1 - ssim(out/10, outputs/10, data_range=1, size_average=True)
            loss_log = log_loss(out/10, outputs/10)
            loss =  0.5 * loss_l1 + 0.8 * loss_ssim
            loss.backward()
            running_loss = running_loss + loss.item()
            optimizer.step()
        a = running_loss / i
        print(f'[{epoch + 1}] training_loss: {running_loss / i :.3f}')
        print('{} seconds'.format(time.time() - t0))
        torch.save(model.state_dict(), PATH_SAVE)
        print('updated training weights')
        scheduler.step()
       

  




if __name__ == '__main__':
    main()