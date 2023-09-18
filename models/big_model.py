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

# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn




class SigLoss(nn.Module):
    """SigLoss.

    Args:
        valid_mask (bool, optional): Whether filter invalid gt
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self,
                 valid_mask=True,
                 loss_weight=1.0,
                 max_depth=None,
                 warm_up=False,
                 warm_iter=100):
        super(SigLoss, self).__init__()
        self.valid_mask = valid_mask
        self.loss_weight = loss_weight
        self.max_depth = max_depth

        self.eps = 0.1 # avoid grad explode

        # HACK: a hack implement for warmup sigloss
        self.warm_up = warm_up
        self.warm_iter = warm_iter
        self.warm_up_counter = 0
        self.rms = torch.nn.MSELoss()

    def sigloss(self, input, target):
        if self.valid_mask:
            valid_mask = target > 0
            if self.max_depth is not None:
                valid_mask = torch.logical_and(target > 0, target <= self.max_depth)
            input = input[valid_mask]
            target = target[valid_mask]
        
        if self.warm_up:
            if self.warm_up_counter < self.warm_iter:
                g = torch.log(input + self.eps) - torch.log(target + self.eps)
                g = 0.15 * torch.pow(torch.mean(g), 2)
                self.warm_up_counter += 1
                return torch.sqrt(g)

        g = torch.log(input) - torch.log(target)
        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        return torch.sqrt(Dg)

    def rmseloss(self, input, target):
        if self.valid_mask:
            valid_mask = target > 0
            if self.max_depth is not None:
                valid_mask = torch.logical_and(target > 0, target <= self.max_depth)
            input = input[valid_mask]
            target = target[valid_mask]

        # return torch.log(torch.sqrt(self.rms(input, target)))
        return torch.sqrt(self.rms(input, target))

    def sqrelloss(self, input, target):
        if self.valid_mask:
            valid_mask = target > 0
            if self.max_depth is not None:
                valid_mask = torch.logical_and(target > 0, target <= self.max_depth)
            input = input[valid_mask]
            target = target[valid_mask]

        return torch.mean(torch.pow(input - target, 2) / target)

    def forward(self,
                depth_pred,
                depth_gt,
                **kwargs):
        """Forward function."""
        
        loss_depth = self.loss_weight * self.sigloss(
            depth_pred,
            depth_gt,
            )

        # loss_depth = self.rmseloss(depth_pred, depth_gt)

        # loss_depth = self.loss_weight * self.sigloss(
        #     depth_pred,
        #     depth_gt,
        #     ) + self.rmseloss(depth_pred, depth_gt)

        # loss_depth = self.sigloss(
        #     depth_pred,
        #     depth_gt,
        #     ) + 2 * self.rmseloss(depth_pred, depth_gt) + 4 * self.sqrelloss(depth_pred, depth_gt)
        return loss_depth


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
    
    self.relu = nn.ReLU(inplace=True)
    self.conv1 = nn.Sequential(
    nn.Conv2d(384, 384, kernel_size, stride=1, padding=1, bias=False),
    #nn.BatchNorm2d(384),
    #nn.ReLU(inplace=True),
)
    self.conv2 = nn.Sequential(
    nn.Conv2d(464, 260, kernel_size, stride=1, padding=1, bias=False),
    #nn.BatchNorm2d(260),
    #nn.ReLU(inplace=True),
)
    self.conv3 = nn.Sequential(
    nn.Conv2d(324, 130, kernel_size, stride=1, padding=1, bias=False),
    #nn.BatchNorm2d(130),
    #nn.ReLU(inplace=True),
)
    self.conv4 = nn.Sequential(
    nn.Conv2d(178, 90, kernel_size, stride=1, padding=1, bias=False),
    #nn.BatchNorm2d(90),
    #nn.ReLU(inplace=True),
)
    self.conv5 = nn.Sequential(
    nn.Conv2d(122, 64, kernel_size, stride=1, padding=1, bias=False),
    #nn.BatchNorm2d(64),
    #nn.ReLU(inplace=True),
    )

  
    
    self.conv6 = nn.Sequential(
    nn.Conv2d(64, 1, 1, 1, 0, bias=False),
    #nn.BatchNorm2d(1),
    nn.Sigmoid()
)

   


  def forward(self,x,dec1,dec2,dec3,dec4):
   
    x = self.conv1(x)
    x1 = self.conv3_19(x)
    x2 = self.conv3_13(x)
    x3 = self.conv3_7(x)
    x = x1 + x2 + x3 + x
    #x = self.b1(x)
    x = self.relu(x)
    x= F.interpolate(x,scale_factor=2, mode= 'nearest')

    #---------------
    x = torch.cat((x,dec4),1)
    x = self.conv2(x)
    x1 = self.conv2_3_19(x)
    x2 = self.conv2_3_13(x)
    x3 = self.conv2_3_7(x)
    x = x1 + x2 + x3 + x
    #x = self.b2(x)
    x = self.relu(x)
    x= F.interpolate(x,scale_factor=2, mode= 'nearest')
     #---------------
    x = torch.cat((x,dec3),1)
    x = self.conv3(x)
    x1 = self.conv3_3_19(x)
    x2 = self.conv3_3_13(x)
    x3 = self.conv3_3_7(x)
    x = x1 + x2 + x3 + x
    #x = self.b3(x)
    x = self.relu(x)
    x= F.interpolate(x,scale_factor=2, mode= 'nearest')
     #---------------
    x = torch.cat((x,dec2),1)
    x = self.conv4(x)
    x1 = self.conv4_3_19(x)
    x2 = self.conv4_3_13(x)
    x3 = self.conv4_3_7(x)
    x = x1 + x2 + x3 + x
   # x = self.b4(x)
    x = self.relu(x)
    x= F.interpolate(x,scale_factor=2, mode= 'nearest')
    #---------------
    x = torch.cat((x,dec1),1)
    x = self.conv5(x) 
    x1 = self.conv5_3_19(x)
    x2 = self.conv5_3_13(x)
    x3 = self.conv5_3_7(x)
    x = x1 + x2 + x3 + x
    #x = self.b5(x)
    x = self.relu(x)
    x= F.interpolate(x,scale_factor=2, mode= 'nearest')
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


