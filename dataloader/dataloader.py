from email.mime import image
import glob
import os
from statistics import mean
from sys import stderr
from turtle import forward
import torch
from PIL import Image, ImageOps
import torchvision
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import math
import random
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def crop_image(image, width, height):
    img_width, img_height = image.size
    left = (img_width - width) // 2
    top = (img_height - height) // 2
    right = left + width
    bottom = top + height
    cropped_image = image.crop((left, top, right, bottom))
    return cropped_image

def inverse_depth_norm(depth):
    zero_mask = depth == 0.0
    depth = 10 / depth
    depth = torch.clamp(depth, 10 / 100, 10)
    depth[zero_mask] = 0.0
    return depth


class NyuDepth_eigen_test(Dataset):

  def __init__(self):
    transformation = transforms.ToTensor()
    self.nyu_depth = pd.read_csv(r'C:\Users\ppapadop\Desktop\nyu2_eigen_test.csv', sep = ',')
    self.meanx = [0.485, 0.456, 0.406]
    self.stdx = [0.229, 0.224, 0.225]

    self.n_samples = self.nyu_depth.shape[0]

  
  def __getitem__(self, index):
      
    file_path =  self.nyu_depth._get_value(index, 'data', takeable=False)
    data = np.load(file_path)
    
    
    for array_name in data.files:
        array_data = data[array_name]

        if array_name == 'image':
            image = array_data
           
        
            
        elif array_name == 'depth':
            depth = array_data
       
    to_PIL = transforms.ToPILImage()    
    #crop_width = 620
    #crop_height = 460
    #image = crop_image(image, crop_width, crop_height)
    #depth = crop_image(depth, crop_width, crop_height)
    image = to_PIL(image)

    x_transforms = transforms.Compose([
       transforms.Resize((448,448)),
       transforms.ToTensor(),
       transforms.Normalize(torch.Tensor(self.meanx), torch.Tensor(self.stdx))
    ])
  
    to_tensor = transforms.ToTensor()

    #depth = np.array(depth).astype(np.float32)
    #depth = depth/1000.0  #From 8bit to range [0, 10] (meter)
    #zero_mask = depth == 0.0
    #depth = to_tensor(depth)
    #depth = torch.tensor(depth).unsqueeze(dim=0)
    #depth = torch.clamp(depth, 10/100.0, 10)
   
    image = x_transforms(image)

    image = renormalize(image)
    return {'image': image, 'depth': depth,}
  
  def __len__(self):
    return self.n_samples


class NyuDepth_train(Dataset):

  def __init__(self):
    self.nyu_depth = pd.read_csv(r'C:\Users\ppapadop\Desktop\depth_vir\datasets\train.csv', sep = ',')
    self.meanx = [0.485, 0.456, 0.406]
    self.stdx = [0.229, 0.224, 0.225]

    self.n_samples = self.nyu_depth.shape[0]

  def __getitem__(self, index):
    path_x = self.nyu_depth._get_value(index, 'image', takeable=False)
    image = Image.open(path_x)
    path_y = self.nyu_depth._get_value(index, 'depth', takeable=False)
    depth = Image.open(path_y)
    crop_width = 620
    crop_height = 460
    image = crop_image(image, crop_width, crop_height)
    depth = crop_image(depth, crop_width, crop_height)

    x_transforms = transforms.Compose([
       transforms.Resize((448,448)),
       transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.3, hue=0.2),
       transforms.ToTensor(),
       transforms.Normalize(torch.Tensor(self.meanx), torch.Tensor(self.stdx))
    ])
    veritcal = transforms.RandomHorizontalFlip(p=1)
   
    if random.random() <0.5 :
      image = veritcal(image)
      depth = veritcal(depth)
     
    
    
    y_transforms =transforms.Compose([ transforms.Resize((448,448))])
    
    depth = y_transforms(depth) 
    transformation = transforms.ToTensor()
    depth = np.array(depth).astype(np.float32)
    depth = depth /255.0 * 10.0 #From 8bit to range [0, 10] (meter)
    zero_mask = depth == 0.0
    depth = transformation(depth)
    depth = torch.clamp(depth, 10/100.0, 10)
    image = x_transforms(image)
    image = renormalize(image)
   
    return {'image': image, 'depth': depth}

  def __len__(self):
    return self.n_samples



def renormalize(tensor):
    minFrom= tensor.min()
    maxFrom= tensor.max()
    minTo = 0
    maxTo=1
    return minTo + (maxTo - minTo) * ((tensor - minFrom) / (maxFrom - minFrom))


def aa(loader) :
  mean = 0.
  std = 0.
  for images, _ in loader:
      print(images[0])
      batch_samples = images.size(0) # batch size (the last batch can have smaller size!)
      images = images.view(batch_samples, images.size(1), -1)
      mean += images.mean(2).sum(0)
      std += images.std(2).sum(0)

  mean /= len(loader.dataset)
  std /= len(loader.dataset)

  return mean ,std


def main() :
   transformss = transforms.ToPILImage()
   train_dataset = NyuDepth_eigen_test()
   train_load = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
   datatiter = iter(train_load)
   data = next(datatiter)
   input = torch.autograd.Variable(data['image'].cuda())
   output = torch.autograd.Variable(data['depth'].cuda())
   print(input.shape)
   print(output.shape)
  


if __name__ == '__main__':
    main()
