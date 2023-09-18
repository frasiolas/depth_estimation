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
import random
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
import glob
from dataloader import NyuDepth_test, NyuDepth_eigen_test

from mobilevit_models.lightweight_model import EncoderDecoder
from test import inverse_depth_norm
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


def main():
    model = EncoderDecoder(batch_size=1)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    transformss = transforms.ToPILImage()
    y_transforms =transforms.Compose([ transforms.Resize((480,640))])
    PATH = r'C:\Users\ppapadop\Desktop\depth_vir\mobilenet_dilation.pth'
    model.load_state_dict(torch.load(PATH))
    model.to(device)
    transformation = transforms.ToTensor()
    
    
    metric_name = ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log',
                'log10', 'silog']
    result_metrics = {}
    for metric in metric_name:
        result_metrics[metric] = 0.0
    dataset = NyuDepth_eigen_test()
    total_samples = len(dataset)
    print(total_samples)
    model.eval()
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, pin_memory=True)
    for i, data in enumerate(dataloader, 0):
        inputs = data['image'].cuda()
        outputs = data['depth'].cuda()
        pred = model.forward(inputs)
        pred= F.interpolate(pred,size = (480,640), mode= 'bilinear', align_corners= True) 
        pred = torch.squeeze(pred, dim=0)
    

        #oo = transformss(pred[0]/10.0)
        #oo.show()
        pred = pred[:, 20:459, 24:615]
        outputs = outputs[:, 20:459, 24:615]
        #transformss(outputs[0]/10).show()
        #transformss(pred[0]/10).show()
        computed_result = eval_depth(pred, outputs)
        for metric in metric_name:
            result_metrics[metric] += computed_result[metric]
    for key in result_metrics.keys():
        result_metrics[key] = result_metrics[key] / (total_samples)
    print(result_metrics)


if __name__ == '__main__':
    main()

