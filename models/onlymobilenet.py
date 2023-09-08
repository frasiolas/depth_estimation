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
from dataloader import  NyuDepth_train, NyuDepth_test,  NyuDepth_ttrain, NyuDepth_testt
import numpy as np
import time
from timm import models as tmod
import torch.nn.functional as F





device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')




class Decoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.activation = torch.nn.ReLU()
    self.sigmoid = torch.nn.Sigmoid()
    self.conv1 = torch.nn.Conv2d(1280,512,kernel_size=3,stride=1, padding=1)
    self.norm1 = nn.BatchNorm2d(512)
    self.conv2 = torch.nn.Conv2d(608,356,kernel_size=3,stride=1, padding=1)
    self.norm2 = nn.BatchNorm2d(356)
    self.conv3 = torch.nn.Conv2d(388,256,kernel_size=3,stride=1, padding=1)
    self.norm3 = nn.BatchNorm2d(256)
    self.conv4 = torch.nn.Conv2d(280,128,kernel_size=3,stride=1, padding=1)
    self.norm4 = nn.BatchNorm2d(128)
    self.conv5 = torch.nn.Conv2d(144,64,kernel_size=3,stride=1, padding=1)
    self.norm5 = nn.BatchNorm2d(64)
    self.conv6 = torch.nn.Conv2d(64,1,kernel_size=3,stride=1, padding=1)


  def forward(self,x,dec1,dec2,dec3,dec4):
    x = self.conv1(x)
    x= self.norm1(x)
    x = self.activation(x)
    x= F.interpolate(x,scale_factor=2, mode= 'bilinear', align_corners= True) #512 14 14

    #---------------
    x = torch.cat((x, dec4), 1) # 608 14 14
    x = self.conv2(x)
    x= self.norm2(x)
    x = self.activation(x)
    x= F.interpolate(x,scale_factor=2, mode= 'bilinear', align_corners= True) #512 14 14
   
     #---------------
    x = torch.cat((x, dec3), 1) # 388 28 28
    x = self.conv3(x)
    x= self.norm3(x)
    x = self.activation(x)
    x= F.interpolate(x,scale_factor=2, mode= 'bilinear', align_corners= True) #512 14 14
    
     #---------------
    x = torch.cat((x, dec2), 1) #280 56 56
    x = self.conv4(x)
    x= self.norm4(x)
    x = self.activation(x)
    x= F.interpolate(x,scale_factor=2, mode= 'bilinear', align_corners= True) #512 14 14
    
    #---------------
    x = torch.cat((x, dec1), 1) # 144 122 122
    x = self.conv5(x)
    x= self.norm5(x)
    x = self.activation(x)
    x= F.interpolate(x,scale_factor=2, mode= 'bilinear', align_corners= True) #512 14 14
    

     #---------------

    x= self.conv6(x)
    x= self.sigmoid(x)*10
    #x= self.activation(x)
   

    return x


class EncoderDecoder(nn.Module):

    def __init__(self,batch_size):
        super().__init__()
        
        self.Encoder = models.mobilenet_v2( pretrained=True )    
        self.Decoder = Decoder()
        self.Up = torch.nn.Upsample(scale_factor=2, mode = 'bilinear' ,align_corners=True)
        self.batch_size = batch_size
    def forward(self,x):
        features = [x]
        for k, v in self.Encoder.features._modules.items(): features.append( v(features[-1]) )
        out1 = features[19]
        dec4 =features[12]
        dec3 =features[6]
        dec2 =features[3]
        dec1 =features[2]
        out = self.Decoder(out1,dec1,dec2,dec3,dec4)
        return out

class Encoder(nn.Module):

    def __init__(self,batch_size):
        super().__init__()
        
        self.Encoder = models.mobilenet_v2( pretrained=True )    

        self.Up = torch.nn.Upsample(scale_factor=2, mode = 'bilinear' ,align_corners=True)
        self.batch_size = batch_size
    def forward(self,x):
        features = [x]
        for k, v in self.Encoder.features._modules.items(): features.append( v(features[-1]) )
        out1 = features[19]
        dec4 =features[12]
        dec3 =features[6]
        dec2 =features[3]
        dec1 =features[2]
        out = self.Decoder(out1,dec1,dec2,dec3,dec4)
        return out



def main() :
 
    transformss = transforms.ToPILImage()
    model = EncoderDecoder(batch_size=8)
    num_epochs = 15
    BatchSize = 8
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)
    PATH_SAVE = r'C:\Users\panagiotis_local\vscode\mobilenet.pth'
    PATH_SAVE_val = r'C:\Users\panagiotis_local\vscode\mobilenet.pth'
    PATH = r'C:\Users\panagiotis_local\vscode\mobilenet.pth'
    model.load_state_dict(torch.load(PATH))
    criterion = nn.MSELoss()
    #criterion = torch.nn.HuberLoss(reduction='mean', delta=1.0)
    optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    train_dataset = NyuDepth_ttrain()
    train_load = DataLoader(dataset=train_dataset, batch_size=BatchSize, shuffle=True)
    validation_dataset = NyuDepth_test()
    validation_load = DataLoader(dataset=validation_dataset, batch_size=BatchSize, shuffle=False)
    model.cuda()
    model.train() 
     
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


        
            #calculate loss
            loss =criterion(out, outputs)
            #l_ssim = torch.clamp((1 - ssim(out, depth_n, val_range = 1000.0 / 10.0)) * 0.5, 0, 1)
            #loss = (1.0 * l_ssim.mean().item()) + (0.1 * l_depth)
            #losses.update(loss.data.item(), image.size(0))
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
        model.eval()
        #model.eval()
        #for i, data in enumerate(validation_load, 0):
            #inputs, outputs = data
            #inputs = inputs.to(device)
            #out = model.forward(inputs)
            #out = out.to(device)
            #outputs = outputs.to(device) 
            #loss = criterion(out, outputs)  
            #val_loss += loss.item()
        #print(f'[{epoch + 1}] validation_loss: {val_loss / i :.3f}')
        #if val_loss < key:
           #key = val_loss
           #torch.save(model.state_dict(), PATH_SAVE_val)
           #print('updated val weights')
        #model.train()
        #with open('l1loss.txt', 'a') as f:
            #f.write(f'[{epoch + 1}] training_loss =: {a :.3f} validation_loss: {val_loss / i :.3f}')
            #f.write('\n')
        #scheduler.step()

  




if __name__ == '__main__':
    main()

