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

from mobilevit_models.simple_model import EncoderDecoder
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


def eval_depth2(pred, target):
    assert pred.shape == target.shape

    thresh = torch.max((target / pred), (pred / target))

    d1 = torch.sum(thresh < 1.25).float() / len(thresh)
    d2 = torch.sum(thresh < 1.25 ** 2).float() / len(thresh)
    d3 = torch.sum(thresh < 1.25 ** 3).float() / len(thresh)

    diff = pred - target
    diff_log = torch.log(pred) - torch.log(target)

    abs_rel = torch.mean(torch.abs(diff) / target)
    sq_rel = torch.mean(torch.pow(diff, 2) / target)

    rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
    rmse_log = torch.sqrt(torch.mean(torch.pow(diff_log , 2)))

    log10 = torch.mean(torch.abs(torch.log10(pred) - torch.log10(target)))
    silog = torch.sqrt(torch.pow(diff_log, 2).mean() - 0.5 * torch.pow(diff_log.mean(), 2))

    return {'d1': d1.item(), 'd2': d2.item(), 'd3': d3.item(), 'abs_rel': abs_rel.item(),
            'sq_rel': sq_rel.item(), 'rmse': rmse.item(), 'rmse_log': rmse_log.item(), 
            'log10':log10.item(), 'silog':silog.item()}

def eval_depth(pred, target):
        pred = torch.squeeze(pred)
        target = torch.squeeze(target)
        assert pred.shape == target.shape

        thresh = torch.max((target / pred), (pred / target))
        
        d1 = torch.sum(thresh < 1.25).float() / (thresh.shape[0] * thresh.shape[1])
        d2 = torch.sum(thresh < 1.25 ** 2).float() /(thresh.shape[0] * thresh.shape[1])
        d3 = torch.sum(thresh < 1.25 ** 3).float() / (thresh.shape[0] * thresh.shape[1])
        diff = pred - target
        diff_log = torch.log(pred) - torch.log(target)
        target[target<0.001] = 0.001
        target[target>10] = 10
        pred[pred<0.001] = 0.001
        pred[pred>10] = 10
        abs_rel = torch.mean(torch.abs(diff) / (target))
        sq_rel = torch.mean(torch.pow(diff, 2) / (target))

        rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
        rmse_log = torch.sqrt(torch.mean(torch.pow(diff_log , 2)))

        log10 = torch.mean(torch.abs(torch.log10(pred) - torch.log10(target)))
        silog = torch.sqrt(torch.pow(diff_log, 2).mean() - 0.5 * torch.pow(diff_log.mean(), 2))

        return {'d1': d1.item(), 'd2': d2.item(), 'd3': d3.item(), 'abs_rel': abs_rel.item(),
                'sq_rel': sq_rel.item(), 'rmse': rmse.item(), 'rmse_log': rmse_log.item(), 
                'log10':log10.item(), 'silog':silog.item()}

def inverse_depth_norm(depth):
    zero_mask = depth == 0.0
    depth = 10 / depth
    depth = torch.clamp(depth, 10 / 100, 10)
    depth[zero_mask] = 0.0
    return depth

def main():

    meanx = [0.485, 0.456, 0.406]
    stdx = [0.229, 0.224, 0.225]

    transformss = transforms.ToPILImage()
    to_tensor =  transforms.ToTensor()
    model = EncoderDecoder(batch_size=1)
    model.cuda()
    PATH = r'C:\Users\ppapadop\Desktop\depth_vir\mobilenet_simple.pth'
    model.load_state_dict(torch.load(PATH))
    model.eval()
 

    arrays = {}
    f = h5py.File( r'C:\Users\ppapadop\Downloads\nyu_depth_v2_labeled.mat')
    for k, v in f.items():
        arrays[k] = np.array(v)
    depth = arrays['depths']
    rgb = arrays['images']
    #depth = torch.tensor(depth)
    rgb = torch.tensor(rgb)

    x_transforms = transforms.Compose([ transforms.Resize((448,448)),
       transforms.ToTensor(),
       transforms.Normalize(torch.Tensor(meanx), torch.Tensor(stdx))
    ])
    crop_width = 620
    crop_height = 460
   


    metric_name = ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log',
                'log10', 'silog']
    result_metrics = {}
    for metric in metric_name:
        result_metrics[metric] = 0.0

 
    model.eval()
    transformss = transforms.ToPILImage()
    for i in range (1449):
        input = rgb[i]
        input = input.permute(0,2,1)
        input = input[:, 10:470, 10:630]
        output = depth[i]
        output = np.array(output).astype(np.float32)
        output = to_tensor(output)
        output = output.permute(0,2,1).cuda()
        output = output[:, 10:470, 10:630]
        input = transformss(input)
        input = x_transforms(input)
        input = renormalize(input).cuda()
        input = torch.unsqueeze(input,0)
        pred = model.forward(input)
        pred= F.interpolate(pred,size = (460,620), mode= 'bilinear', align_corners= True) 
        #transformss(output/10).show()
        #transformss(pred[0]/10).show()

        
        #pred = inverse_depth_norm(pred)
        pred = torch.squeeze(pred)
        output= torch.squeeze(output)
        computed_result = eval_depth(pred, output)
        for metric in metric_name:
            result_metrics[metric] += computed_result[metric]
    for key in result_metrics.keys():
        result_metrics[key] = result_metrics[key] / i
    print(result_metrics)


if __name__ == '__main__':
    main()

