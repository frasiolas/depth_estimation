from email.mime import image
import glob
import os
from pickle import TRUE
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
import numpy as np
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
from torch.utils.data import Dataset, dataloader
from dataloader import NyuDepth_train, NyuDepth_test
import numpy as np
from dataloader import renormalize
import matplotlib.pyplot as plt
import h5py

from mobilevit_models.model_with_dilation import EncoderDecoder
from io import BytesIO
from dataloader import renormalize
from test import inverse_depth_norm

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import os
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
#from mmseg.models.builder import BACKBONES
#from mmseg.utils import get_root_logger
#from mmcv.runner import load_checkpoint
import random
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
import glob
import torchsummary


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


def renormalize(tensor):
    minFrom= tensor.min()
    maxFrom= tensor.max()
    minTo = 0
    maxTo=1
    return minTo + (maxTo - minTo) * ((tensor - minFrom) / (maxFrom - minFrom))


def main():

    meanx = [0.485, 0.456, 0.406]
    stdx = [0.229, 0.224, 0.225]
    path_x = r'D:\1234.jpg'
    input = Image.open(path_x)

    transformss = transforms.ToPILImage()
    to_tensor =  transforms.ToTensor()
    model = EncoderDecoder(batch_size=1)
    model.cuda()
    PATH = r'C:\Users\ppapadop\Desktop\depth_vir\mobilenet_dilation.pth'
    model.load_state_dict(torch.load(PATH))
    model.eval()
    #crop_width = 620
    #crop_height = 460
    #input = crop_image(input, crop_width, crop_height)
   

    x_transforms = transforms.Compose([
       transforms.Resize((448,448)),
       transforms.ToTensor(),
       transforms.Normalize(torch.Tensor(meanx), torch.Tensor(stdx))
    ])
   

    input = x_transforms(input)
    input = renormalize(input)
    input = torch.unsqueeze(input,0).cuda()
    output = model.forward(input)
    input= F.interpolate(input,size = (460,620), mode= 'bilinear', align_corners= True) 
    output= F.interpolate(output,size = (460,620), mode= 'bilinear', align_corners= True) 
    transformss(input[0]).show()
    transformss(output[0]/10.0).show()

   
    


if __name__ == '__main__':
    main()

