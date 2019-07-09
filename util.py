# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 21:12:33 2019

@author: tungo
"""

from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2 

def process_prediction(x, dims, anchors, num_classes, CUDA):
    num_anchors = len(anchors)
    fmap_size = x.size(2)
    height, width = dims[0], dims[1]
    scale_h, scale_w = height // x.size(2), width // x.size(3)
    
    # transform x
    x = x.view(x.size(0), x.size(1), -1)
    x = x.transpose(1, 2).contiguous()
    x = x.view(x.size(0), x.size(1)*num_anchors, -1)
    
    # scale anchors' dims with respect to feature map
    anchors = [(a[0]//scale_h, a[1]//scale_w) for a in anchors]
    
    # calculate boxes' centers and objectness
    x[:,:,0] = torch.sigmoid(x[:,:,0])
    x[:,:,1] = torch.sigmoid(x[:,:,1])
    x[:,:,4] = torch.sigmoid(x[:,:,4])
    
    # Add the center offsets
    grid = np.arange(fmap_size)  # height = width -> pick one
    a,b = np.meshgrid(grid, grid)
    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)
    x[:,:,:2] += x_y_offset

    # Calculate boxes' height and width
    anchors = torch.FloatTensor(anchors)
    anchors = anchors.repeat(fmap_size*fmap_size, 1).unsqueeze(0)
    x[:,:,2:4] = torch.exp(x[:,:,2:4])*anchors
    
    # Calculate class score
    x[:,:,5:] = torch.sigmoid(x[:,:,5:])
    
    # Scale back to normal size
    x[:,:,:4] *= scale_h
    return x