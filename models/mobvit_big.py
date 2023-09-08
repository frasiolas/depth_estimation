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
import numpy as np
import time
from timm import models as tmod
from torchsummary import summary
from pytorch_msssim import ms_ssim, ssim
import torch
import torch.nn.functional as F

from math import exp

class Depth_Loss():
    def __init__(self, alpha, beta, gamma, maxDepth=10.0):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.maxDepth = maxDepth

        self.L1_Loss = torch.nn.L1Loss()


    def __call__(self, output, depth):
        if self.beta == 0 and self.gamma == 0:
            valid_mask = depth>0.0
            output = output[valid_mask]
            depth = depth[valid_mask]
            l_depth = self.L1_Loss(output, depth)
            loss = l_depth
        else:
            l_depth = self.L1_Loss(output, depth)
            l_ssim = torch.clamp((1-self.ssim(output, depth, self.maxDepth)) * 0.5, 0, 1)
            l_grad = self.gradient_loss(output, depth)

            loss = self.alpha * l_depth + self.beta * l_ssim + self.gamma * l_grad
        return loss


    def ssim(self, img1, img2, val_range, window_size=11, window=None, size_average=True, full=False):
        L = val_range

        padd = 0
        (_, channel, height, width) = img1.size()
        if window is None:
            real_size = min(window_size, height, width)
            window = self.create_window(real_size, channel=channel).to(img1.device)
            padd = window_size // 2

        mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
        mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

        C1 = (0.01 * L) ** 2
        C2 = (0.03 * L) ** 2

        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        cs = torch.mean(v1 / v2)  # contrast sensitivity

        ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

        if size_average:
            ret = ssim_map.mean()
        else:
            ret = ssim_map.mean(1).mean(1).mean(1)

        if full:
            return ret, cs

        return ret


    def gradient_loss(self, gen_frames, gt_frames, alpha=1):
        gen_dx, gen_dy = self.gradient(gen_frames)
        gt_dx, gt_dy = self.gradient(gt_frames)

        grad_diff_x = torch.abs(gt_dx - gen_dx)
        grad_diff_y = torch.abs(gt_dy - gen_dy)

        # condense into one tensor and avg
        grad_comb = grad_diff_x ** alpha + grad_diff_y ** alpha

        return torch.mean(grad_comb)


    def gradient(self, x):
        """
        idea from tf.image.image_gradients(image)
        https://github.com/tensorflow/tensorflow/blob/r2.1/tensorflow/python/ops/image_ops_impl.py#L3441-L3512
        """
        h_x = x.size()[-2]
        w_x = x.size()[-1]

        left = x
        right = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
        top = x
        bottom = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]

        dx, dy = right - left, bottom - top

        # dx will always have zeros in the last column, right-left
        # dy will always have zeros in the last row,    bottom-top
        dx[:, :, :, -1] = 0
        dy[:, :, -1, :] = 0

        return dx, dy


    def create_window(self, window_size, channel=1):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window


    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()


  

def depthwise(in_channels, kernel_size):
    padding = (kernel_size-1) // 2
    assert 2*padding == kernel_size-1, "parameters incorrect. kernel={}, padding={}".format(kernel_size, padding)
    return nn.Sequential(
          nn.Conv2d(in_channels,in_channels,kernel_size,stride=1,padding=padding,bias=False,groups=in_channels),
          #nn.BatchNorm2d(in_channels),
          nn.ReLU(inplace=True),
        )

def pointwise(in_channels, out_channels):
    return nn.Sequential(
          nn.Conv2d(in_channels,out_channels,1,1,0,bias=False),
          #nn.BatchNorm2d(out_channels),
          nn.ReLU(inplace=True),
        )
def pointwise_last(in_channels, out_channels):
    return nn.Sequential(
          nn.Conv2d(in_channels,out_channels,1,1,0,bias=False),
          #nn.BatchNorm2d(out_channels),
          nn.Sigmoid(),
        )




class Decoder(nn.Module):
  def __init__(self):
    super().__init__()
    #self.activation = torch.nn.ReLU()
    self.sigmoid = torch.nn.Sigmoid()
    kernel_size = 3
    self.conv1 = nn.Sequential(
                depthwise(640, kernel_size),
                pointwise(640, 640))
    self.conv2 = nn.Sequential(
                depthwise(768, kernel_size),
                pointwise(768, 384))
    self.conv3 = nn.Sequential(
                depthwise(480, kernel_size),
                pointwise(480, 240))
    self.conv4 = nn.Sequential(
                depthwise(304, kernel_size),
                pointwise(304, 152))
    self.conv5 = nn.Sequential(
                depthwise(184, kernel_size),
                pointwise(184, 64))
    self.conv6 = pointwise(64, 1)


  def forward(self,x,dec1,dec2,dec3,dec4):
   
    x = self.conv1(x)
    x= F.interpolate(x,scale_factor=2, mode= 'bilinear', align_corners= True) 
    #---------------
    x = torch.cat((x,dec4),1)
    x = self.conv2(x)
    x= F.interpolate(x,scale_factor=2, mode= 'bilinear', align_corners= True)
     #---------------
    x = torch.cat((x,dec3),1)
    x = self.conv3(x)
    x= F.interpolate(x,scale_factor=2, mode= 'bilinear', align_corners= True) 
     #---------------
    x = torch.cat((x,dec2),1)
    x = self.conv4(x)
    x= F.interpolate(x,scale_factor=2, mode= 'bilinear', align_corners= True) 
    #---------------
    x = torch.cat((x,dec1),1)
    x = self.conv5(x)
    x= F.interpolate(x,scale_factor=2, mode= 'bilinear', align_corners= True) 
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
        self.mvit = tmod.create_model('mobilevit_s', pretrained=True, num_classes=0, global_pool='')   
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
 
    num_epochs = 100
    BatchSize = 4
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)
    PATH_SAVE = r'C:\Users\ppapadop\Desktop\vir_env\mobilenet.pth'
    #model.load_state_dict(torch.load(PATH_SAVE))
    criterion = nn.L1Loss()
    
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    train_dataset = NyuDepth_train()
    train_load = DataLoader(dataset=train_dataset, batch_size=BatchSize, shuffle=True)
    
    model.cuda()
    model.train() 
    for epoch in range(num_epochs):  # loop over the dataset multiple times   
        t0 = time.time()
        running_loss = 0.0
        val_loss = 0.0
        for param_group in optimizer.param_groups:
            print(param_group['lr'])
        for i, data in enumerate(train_load, 0):
            optimizer.zero_grad()
            inputs = data['image'].to(device)
            outputs = data['depth'].to(device)
            out = model.forward(inputs)
            out = out.to(device)
            
            loss_l1 =criterion(out, outputs)
            loss_ssim = 1 - ssim(out/10, outputs/10, data_range=1, size_average=True)
            loss =0.7 *loss_l1 + 0.3 * loss_ssim
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