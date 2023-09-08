
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
import time
from timm import models as tmod
from torchsummary import summary

from pytorch_msssim import ms_ssim, ssim
import cv2

def inverse_depth_norm(depth):
    zero_mask = depth == 0.0
    depth = 10 / depth
    depth = torch.clamp(depth, 10 / 100, 10)
    depth[zero_mask] = 0.0
    return depth


def depthwise(in_channels, kernel_size):
    padding = (kernel_size-1) // 2
    assert 2*padding == kernel_size-1, "parameters incorrect. kernel={}, padding={}".format(kernel_size, padding)
    return nn.Sequential(
          nn.Conv2d(in_channels,in_channels,kernel_size,stride=1,padding=padding,bias=False,groups=in_channels),
          nn.LeakyReLU(inplace=True),
        )

def pointwise(in_channels, out_channels):
    return nn.Sequential(
          nn.Conv2d(in_channels,out_channels,1,1,0,bias=False),
           nn.LeakyReLU(inplace=True),
        )

def pointwise_out(in_channels, out_channels):
    return nn.Sequential(
          nn.Conv2d(in_channels,out_channels,1,1,0,bias=False),
           nn.ReLU(inplace=True),
        )



class Decoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.sigmoid = torch.nn.Sigmoid()
    kernel_size = 3
    self.conv1 = nn.Sequential(
                depthwise(320, kernel_size),
                pointwise(320, 240))
    
    self.inter1 = nn.Sequential(
                depthwise(240, kernel_size),
                pointwise(240, 120))
    
    self.inter2 = nn.Sequential(
                depthwise(120, kernel_size),
                pointwise(120, 64))
    self.conv1_plus = nn.Sequential(
                depthwise(64, kernel_size),
                pointwise(64, 64))
    self.conv2 = nn.Sequential(
                depthwise(64, kernel_size),
                pointwise(64, 48))
    self.conv2_plus = nn.Sequential(
                depthwise(48, kernel_size),
                pointwise(48, 48))
    self.conv3 = nn.Sequential(
                depthwise(48, kernel_size),
                pointwise(48, 24))
    self.conv3_plus = nn.Sequential(
                depthwise(24, kernel_size),
                pointwise(24, 24))
    self.conv4 = nn.Sequential(
                depthwise(24, kernel_size),
                pointwise(24, 16))
    self.conv4_plus = nn.Sequential(
                depthwise(16, kernel_size),
                pointwise(16, 16))
    self.conv5 = nn.Sequential(
                depthwise(16, kernel_size),
                pointwise(16, 3))
    self.conv5_plus = nn.Sequential(
                depthwise(3, kernel_size),
                pointwise(3, 3))
    self.conv6 = pointwise_out(3, 1)


  def forward(self,x,dec1,dec2,dec3,dec4):
    
    x = self.conv1(x)
    x = self.inter1(x)
    x = self.inter2(x)
    x= F.interpolate(x,scale_factor=2, mode= 'bilinear', align_corners= True)
    #x = torch.cat([x, x], dim=1)
    x = self.conv1_plus(x)
  
    #---------------
    x = x + dec4
    x = self.conv2(x)
    x= F.interpolate(x,scale_factor=2, mode= 'bilinear', align_corners= True)
    x = self.conv2_plus(x)
   
     #---------------
    x = x + dec3
    x = self.conv3(x)
    x= F.interpolate(x,scale_factor=2, mode= 'bilinear', align_corners= True) 
    x = self.conv3_plus(x)
    
     #---------------
    x = x + dec2
    x = self.conv4(x)
    x= F.interpolate(x,scale_factor=2, mode= 'bilinear', align_corners= True) 
    x = self.conv4_plus(x)
    #---------------
    x = x + dec1
    x = self.conv5(x)
    x= F.interpolate(x,scale_factor=2, mode= 'bilinear', align_corners= True) 
    x = self.conv5_plus(x)
     #--------------
    x= self.conv6(x) 
   
    

    return x



    
features = {}
def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook

class EncoderDecoder(nn.Module):

    def __init__(self,batch_size):
        super().__init__()
        
        self.mvit = tmod.create_model('mobilevit_xxs', pretrained=True, num_classes=0, global_pool='')   
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
    model = EncoderDecoder(batch_size=8)
    total_params = sum(p.numel() for p in model.Decoder.parameters())
    print(f"Total parameters in the decoder: {total_params}")
    num_epochs = 100
    BatchSize = 8
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)
    PATH_SAVE = r'C:\Users\ppapadop\Desktop\vir_env\mobilenet.pth'
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)
    train_dataset = NyuDepth_train()
    train_load = DataLoader(dataset=train_dataset, batch_size=BatchSize, shuffle=True, num_workers=4)
    model.cuda()
    model.train() 
    for epoch in range(num_epochs):  # loop over the dataset multiple times   
        t0 = time.time()
        running_loss = 0.0
        for param_group in optimizer.param_groups:
            print(param_group['lr'])
        for i, data in enumerate(train_load, 0):
            optimizer.zero_grad()
            inputs = data['image'].to(device)
            outputs = data['depth'].to(device)
            
            out = model.forward(inputs)
            out = out.to(device)
            print(outputs.max())
            print(out.max())
            loss_l1 =criterion(out, outputs)
            ssim_out = inverse_depth_norm(out)
            ssim_out = ssim_out/10.0
            ssim_outputs = inverse_depth_norm(outputs)
            ssim_outputs = ssim_outputs/10.0
            loss_ssim = 1 - ssim(ssim_out, ssim_outputs, data_range=1, size_average=True)
            loss = 0.7 *loss_l1 + 0.3 * loss_ssim 
            loss.backward()
          
            #print(loss)
            running_loss = running_loss + loss.item()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

            optimizer.step()
        
        a = running_loss / i
        print(f'[{epoch + 1}] training_loss: {running_loss / i :.3f}')
        print('{} seconds'.format(time.time() - t0))
        torch.save(model.state_dict(), PATH_SAVE)
        print('updated training weights')
        scheduler.step()

     

  




if __name__ == '__main__':
    main()