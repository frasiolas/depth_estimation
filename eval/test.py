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
from dataloader import NyuDepth_test, NyuDepth_train, NyuDepth_eigen_test
import numpy as np
import matplotlib.pyplot as plt

from mobilevit_models.model_with_dilation import EncoderDecoder

def inverse_depth_norm(depth):
    zero_mask = depth == 0.0
    depth = 10 / depth
    depth = torch.clamp(depth, 10 / 100, 10)
    depth[zero_mask] = 0.0
    return depth

def main():
   
    transformss = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()
    model = EncoderDecoder(batch_size=1)
    model.cuda()
    PATH = r'C:\Users\ppapadop\Desktop\depth_vir\mobilenet_dilation.pth'
    model.load_state_dict(torch.load(PATH))
    model.eval()
    dataset = NyuDepth_eigen_test()
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    datatiter = iter(dataloader)
    data = next(datatiter)
    input = torch.autograd.Variable(data['image'].cuda())
    output = torch.autograd.Variable(data['depth'].cuda())
    #rgb = torch.autograd.Variable(data['rgb'].cuda())
    outt = model.forward(input)
    #output = inverse_depth_norm(output)
    #outt = inverse_depth_norm(outt)

    output = output/10.0
    outt = outt/10.0
    output = torch.unsqueeze(output, 0)
    #outt= F.interpolate(outt,size = (460,620), mode= 'nearest',) 
    output= F.interpolate(output,size = (460,620), mode= 'nearest') 
    outt= F.interpolate(outt,size = (460,620), mode= 'nearest') 
    #rgb = transformss(rgb[0])
    #rgb.show()
    oo = transformss(outt[0])
    aa = transformss(output[0])
    aa.show()
    oo.show()
    #output=output.detach().cpu().numpy()
    #output = output.reshape((460,620))
    #fig=plt.figure()
    #fig.set_figheight(20)
    #fig.set_figwidth(20)
    #fig.add_subplot(1,2,1)
    #plt.imshow(output)
    #x=outt.detach().cpu().numpy()
    #x= x.reshape((460,620))

    #fig.add_subplot(1,2,2)
    #plt.imshow(x)
    #plt.show()
    #output =np.expand_dims(output, axis=0)
   
    #x = np.expand_dims(x, axis=0)
   
 

   
 
   

if __name__ == '__main__':
    main()