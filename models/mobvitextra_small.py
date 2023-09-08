
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
from mobvitextra_small_conv import EncoderDecoder1





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

# Load pre-trained VGG model
vgg_model = models.vgg19(weights='IMAGENET1K_V1')
vgg_model = vgg_model.features
vgg_model = nn.Sequential(*list(vgg_model.children())[:35])  # Choose appropriate layers for perceptual loss
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vgg_model = vgg_model.to(device)

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.criterion = nn.L1Loss()  # L1 loss for pixel-level similarity

    def forward(self, input, target):
        input_features = vgg_model(input)
        target_features = vgg_model(target)
        loss = self.criterion(input_features, target_features)
        return loss


class TVNorm(nn.Module):
    def __init__(self):
        super(TVNorm, self).__init__()

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        img = x.view(batch_size, channels, height, width)

        # Compute horizontal differences
        h_diff = img[:, :, :, :-1] - img[:, :, :, 1:]
        h_norm = torch.norm(h_diff, p=2, dim=1)

        # Compute vertical differences
        v_diff = img[:, :, :-1, :] - img[:, :, 1:, :]
        v_norm = torch.norm(v_diff, p=2, dim=1)

        # Compute TV norm
        tv_norm = torch.mean(h_norm) + torch.mean(v_norm)

        return tv_norm


tv_norm = TVNorm()




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

    kernel_size = 3
    self.conv1 = nn.Sequential(
                depthwise(320, kernel_size),
                pointwise(320, 100))
    
    self.plus_conv1 = nn.Sequential(
                depthwise(100, kernel_size),
                pointwise(100, 100))
        
    self.conv2 = nn.Sequential(
                depthwise(164, kernel_size),
                pointwise(164, 52))
    
    self.plus_conv2 = nn.Sequential(
                depthwise(52, kernel_size),
                pointwise(52, 52))
    self.conv3 = nn.Sequential(
                depthwise(100, kernel_size),
                pointwise(100, 24))
    self.plus_conv3 = nn.Sequential(
                depthwise(24, kernel_size),
                pointwise(24, 24))
    self.conv4 = nn.Sequential(
                depthwise(48, kernel_size),
                pointwise(48, 10))
    self.plus_conv4 = nn.Sequential(
                depthwise(10, kernel_size),
                pointwise(10, 10))
    self.conv5 = nn.Sequential(
                depthwise(26, kernel_size),
                pointwise(26, 3))
    self.conv6 = pointwise_out(3, 1)


  def forward(self,x,dec1,dec2,dec3,dec4):

    x = self.conv1(x)
    x= F.interpolate(x,scale_factor=2, mode= 'bilinear', align_corners= True)
    x = self.plus_conv1(x)
    x = self.plus_conv1(x)
    #---------------
    x = torch.cat((x,dec4),1)
    x = self.conv2(x)
    x= F.interpolate(x,scale_factor=2, mode= 'bilinear', align_corners= True)
    x = self.plus_conv2(x)
    x = self.plus_conv2(x)
     #---------------
    x = torch.cat((x,dec3),1)
    x = self.conv3(x)
    x= F.interpolate(x,scale_factor=2, mode= 'bilinear', align_corners= True) 
    x= self.plus_conv3(x)
    x = self.plus_conv3(x)
     #---------------
    x = torch.cat((x,dec2),1)
    x = self.conv4(x)
    x= F.interpolate(x,scale_factor=2, mode= 'bilinear', align_corners= True) 
    x = self.plus_conv4(x)
    x = self.plus_conv4(x)
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
        # Load an example image
    image = Image.open(r'D:\data\nyu2_train\home_office_0011_out\69.jpg')  
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
])
    image = image_transform(image).unsqueeze(0)
    model = EncoderDecoder(batch_size=1)
    PATH_SAVE = r'C:\Users\ppapadop\Desktop\vir_env\mobilenet.pth'
    #model.load_state_dict(torch.load(PATH_SAVE))
# Set the model to evaluation mode
    model.eval()


    # Forward pass through the model
    with torch.no_grad():
        output = model(image)

    # Retrieve the encoder outputs
    stage0_output = features['stage_3']
    for i in range(stage0_output.size(1)):  # Iterate over the channels
        channel_image = stage0_output[:, i, :, :].squeeze().cpu().numpy()
        # Normalize the channel image to the range [0, 1]
        channel_image = (channel_image - channel_image.min()) / (channel_image.max() - channel_image.min())
        # Print the channel image
        plt.imshow(channel_image, cmap='gray')
        plt.title(f"Stage 0 Channel {i}")
        plt.show()


    

  




if __name__ == '__main__':
    main()