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

import matplotlib.pyplot as plt
from pytorch_msssim import ms_ssim, ssim
import cv2


def gradient_loss(predicted_depth, ground_truth_depth):
    # Calculate gradients of predicted and ground truth depth maps
    gradient_predicted = torch.abs(F.conv2d(predicted_depth, weight=torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).unsqueeze(0).unsqueeze(0).to(predicted_depth.device), padding=1))
    gradient_ground_truth = torch.abs(F.conv2d(ground_truth_depth, weight=torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).unsqueeze(0).unsqueeze(0).to(ground_truth_depth.device), padding=1))

    # Compute the mean squared difference between the gradient magnitudes
    loss = F.mse_loss(gradient_predicted, gradient_ground_truth)

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

class SilogLoss(nn.Module):
    def __init__(self):
        super(SilogLoss, self).__init__()

    def forward(self, predicted_depth, ground_truth_depth):
        diff = torch.log(predicted_depth) - torch.log(ground_truth_depth)
        loss = torch.mean(diff**2) - 0.5 * (torch.mean(diff)**2)
        return loss

