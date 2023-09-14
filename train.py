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



def main() :
    

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    transformss = transforms.ToPILImage()
    model = EncoderDecoder(batch_size=4)
    num_epochs = 20
    BatchSize = 8
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)
    PATH_SAVE = r'C:\Users\ppapadop\Desktop\depth_vir\mobilenet_dilation.pth'
    #model.load_state_dict(torch.load(PATH_SAVE))
    criterion = nn.L1Loss()
    log_loss = SigLoss()
   
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay= 0.00001)
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
            #loss_l1 =criterion(out/10, outputs/10)
            #loss_ssim = 1 - ssim(out/10, outputs/10, data_range=1, size_average=True)
            #loss =  0.5 * loss_l1 + 0.5 * loss_ssim
            loss_log = log_loss(out/10, outputs/10)
            loss = loss_log
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
