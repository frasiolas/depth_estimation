
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
import matplotlib.pyplot as plt
from pytorch_msssim import ms_ssim, ssim
import cv2



def inverse_depth_norm(depth):
    zero_mask = depth == 0.0
    depth = 10 / depth
    depth = torch.clamp(depth, 10 / 100, 10)
    depth[zero_mask] = 0.0
    return depth

def gradient_loss(predicted_depth, ground_truth_depth):
    # Calculate gradients of predicted and ground truth depth maps
    gradient_predicted = torch.abs(F.conv2d(predicted_depth, weight=torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).unsqueeze(0).unsqueeze(0).to(predicted_depth.device), padding=1))
    gradient_ground_truth = torch.abs(F.conv2d(ground_truth_depth, weight=torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).unsqueeze(0).unsqueeze(0).to(ground_truth_depth.device), padding=1))

    # Compute the mean squared difference between the gradient magnitudes
    loss = F.mse_loss(gradient_predicted, gradient_ground_truth)

    return loss


def depthwise(in_channels, kernel_size):
    padding = (kernel_size-1) // 2
    assert 2*padding == kernel_size-1, "parameters incorrect. kernel={}, padding={}".format(kernel_size, padding)
    return nn.Sequential(
          nn.Conv2d(in_channels,in_channels,kernel_size,stride=1,padding=padding,bias=False,groups=in_channels),
          nn.ReLU(inplace=True),
        )

def pointwise(in_channels, out_channels):
    return nn.Sequential(
          nn.Conv2d(in_channels,out_channels,1,1,0,bias=False),
           nn.ReLU(inplace=True),
        )

def pointwise_out(in_channels, out_channels):
    return nn.Sequential(
          nn.Conv2d(in_channels,out_channels,1,1,0,bias=False),
           nn.ReLU(inplace=True),
        )

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        kernel_size = 3
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(320, 100, kernel_size=kernel_size, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(100)  # Batch normalization before activation
        self.plus_conv1 = nn.Conv2d(100, 100, kernel_size=kernel_size, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(100)  # Batch normalization before activation
        self.conv2 = nn.Conv2d(164, 52, kernel_size=kernel_size, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(52)  # Batch normalization before activation
        self.plus_conv2 = nn.Conv2d(52, 52, kernel_size=kernel_size, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(52)  # Batch normalization before activation
        self.conv3 = nn.Conv2d(100, 24, kernel_size=kernel_size, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(24)  # Batch normalization before activation
        self.plus_conv3 = nn.Conv2d(24, 24, kernel_size=kernel_size, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(24)  # Batch normalization before activation
        self.conv4 = nn.Conv2d(48, 10, kernel_size=kernel_size, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(10)  # Batch normalization before activation
        self.plus_conv4 = nn.Conv2d(10, 10, kernel_size=kernel_size, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(10)  # Batch normalization before activation
        self.conv5 = nn.Conv2d(26, 3, kernel_size=kernel_size, stride=1, padding=1)
        self.bn9 = nn.BatchNorm2d(3)  # Batch normalization before activation
        self.conv6 = nn.Conv2d(3, 1, kernel_size=kernel_size, stride=1, padding=1)
    
    def forward(self, x, dec1, dec2, dec3, dec4):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.plus_conv1(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.plus_conv1(x)
        x = self.bn2(x)
        x = self.sigmoid(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat((x, dec4), 1)
        x = self.conv2(x)
        x = self.bn3(x)
        x = self.activation(x)
        x = self.plus_conv2(x)
        x = self.bn4(x)
        x = self.activation(x)
        x = self.plus_conv2(x)
        x = self.bn4(x)
        x = self.sigmoid(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat((x, dec3), 1)
        x = self.conv3(x)
        x = self.bn5(x)
        x = self.activation(x)
        x = self.plus_conv3(x)
        x = self.bn6(x)
        x = self.activation(x)
        x = self.plus_conv3(x)
        x = self.bn6(x)
        x = self.sigmoid(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat((x, dec2), 1)
        x = self.conv4(x)
        x = self.bn7(x)
        x = self.activation(x)
        x = self.plus_conv4(x)
        x = self.bn8(x)
        x = self.activation(x)
        x = self.plus_conv4(x)
        x = self.bn8(x)
        x = self.sigmoid(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat((x, dec1), 1)
        x = self.conv5(x)
        x = self.bn9(x)
        x = self.activation(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv6(x)
      
        x = self.sigmoid(x) * 10

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
        stage2 = self.mvit.stages[2].register_forward_hook(get_features('stage_2'))
        stage3 = self.mvit.stages[3].register_forward_hook(get_features('stage_3'))
        self.batch_size = batch_size
        #for param in self.mvit.parameters():
            #param.requires_grad = False
    

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
    BatchSize = 2
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)
    PATH_SAVE = r'C:\Users\ppapadop\Desktop\vir_env\mobilenet.pth'
    #model.load_state_dict(torch.load(PATH_SAVE))
    criterion = nn.MSELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
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
            #print('outputs....',outputs.max())
            #print('out........', out.max())
            loss_l1 =criterion(out, outputs)
            loss_ssim = 1 - ms_ssim(out/10, outputs/10, data_range=1, size_average=True)
            loss = 0.7 * loss_l1 + 0.3 * loss_ssim
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