#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 22:43:38 2019

@author: anhtu
"""

from __future__ import division

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import cv2
from util import *

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()
        
class YoloLayer(nn.Module):
    def __init__(self, anchors):
        super(YoloLayer, self).__init__()
        self.anchors = anchors

def parse_cfg(filename):
    """
    Inputs:
        - cfg's file name, e.g. 'yolov3.cfg'
    
    Returns:
        - a list of NN blocks, each block is represented as a dictionary
    """
    file = open(filename, 'r')
    lines = file.read().split('\n')
    lines = [x.rstrip().lstrip() for x in lines if len(x) > 0 and x[0] != '#']
    
    blocks = []
    block = {}
    for line in lines:
        if line[0] == '[':
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block['type'] = line[1:-1].rstrip()
        else:
            s = line.split('=')
            block[s[0].lstrip().rstrip()] = s[1].lstrip().rstrip()
    blocks.append(block)
    return blocks

def create_modules(blocks):
    net_info = blocks[0]  # [net] contains the info of the entire network
    module_list = nn.ModuleList()
    prev_filters = 3  # initialized with first number of channels (3 - R, G, B)
    output_filters = []
    
    for idx, layer in enumerate(blocks[1:]):
        module = nn.Sequential()
        # CONV layer
        if layer['type'] == 'convolutional':
            activation = layer['activation']
            try:
                batchnorm = int(layer['batch_normalize'])
            except:
                batchnorm = 0
            filters = int(layer['filters'])
            kernel_size = int(layer['size'])
            stride = int(layer['stride'])
            pad = int(layer['pad'])
            padding = None
            
            # pad & padding are different
            if pad == 0:
                padding = 0
            else:
                padding = kernel_size // 2
            
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, padding)
            module.add_module('conv_{}'.format(idx), conv)
            if batchnorm:
                bn = nn.BatchNorm2d(filters)
                module.add_module('batch_norm_{}'.format(idx), bn)

            if activation == 'leaky':
                leaky = nn.LeakyReLU(0.1) # 0.1 according to YOLOv1 paper
                module.add_module('leaky_{}'.format(idx), leaky)
                
        # Upsample layer
        elif layer['type'] == 'upsample':
            stride = int(layer['stride'])
            upsample = nn.Upsample(scale_factor=2, mode='bilinear')
            module.add_module('upsample_{}'.format(idx), upsample)
            
        
        # Concatenation layer
        elif layer['type'] == 'route':
            layer['layers'] = layer['layers'].split(',')
            start_layer = int(layer['layers'][0])
            try:
                end_layer = int(layer['layers'][1])
            except:
                end_layer = 0
            route = EmptyLayer()
            module.add_module('route_{}'.format(idx), route)
            if end_layer == 0:
                filters = output_filters[start_layer + idx]
            else:
                filters = output_filters[start_layer + idx] + output_filters[end_layer]
                
        # Shortcut layer (skip connection)
        elif layer['type'] == 'shortcut':
            shortcut = EmptyLayer()
            module.add_module('shortcut_{}'.format(idx), shortcut)
            
        # YOLO layer
        elif layer["type"] == "yolo":
            mask = layer["mask"].split(",")
            mask = [int(x) for x in mask]

            anchors = layer["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]
            anchors = [anchors[i] for i in mask]

            detection = YoloLayer(anchors)
            module.add_module("Detection_{}".format(idx), detection)
        
        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
    return (net_info, module_list)
            
# create network
class Net(nn.Module):
    def __init__(self, filename):
        super(Net, self).__init__()
        self.blocks = parse_cfg(filename)
        self.net_info, self.module_list = create_modules(self.blocks)
    
    def __str__(self):
        return ('** Information about the network: ' + str(self.net_info) + '\n\n' +
                '** All layers of the network: \n' + str(self.module_list))
    def forward(self, x, CUDA):
        layers = self.blocks[1:] # except the 'net' module
        outputs = {}
        yolo_calc = 0
        for idx, layer in enumerate(layers):
            if layer['type'] == 'convolutional' or layer['type'] == 'upsample':
                x = self.module_list[idx](x)
            elif layer['type'] == 'route':
                l = [int(x) for x in layer['layers']]
                if len(l) == 1:
                    x = outputs[idx + l[0]]
                else:
                    out1 = outputs[idx + l[0]]
                    out2 = outputs[l[1]]
                    x = torch.cat([out1, out2], dim=1)
            elif layer['type'] == 'shortcut':
                x = outputs[int(layer['from'])+idx] + outputs[idx-1]
            elif layer['type'] == 'yolo':
                anchors = self.module_list[idx][0].anchors
                inp_dims = (int(self.net_info['height']), int(self.net_info['width']))
                num_classes = int(layer['classes'])
                # x has shape (batch_size, (4+1+80)*3, N, N)
                # in which, 4: bbox offsets, 1: objectness score, 80: classes, 3: num of boxes, N: box's dimension
                x = process_prediction(x, inp_dims, anchors, num_classes, CUDA)
                if not yolo_calc:              #if no collector has been intialised. 
                    detections = x
                    yolo_calc = 1
                else:       
                    detections = torch.cat((detections, x), 1)
            outputs[idx] = x
        return detections
    
def get_test_input():
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (416,416))          #Resize to the input dimension
    img_ =  img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W 
    img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()     #Convert to float
    img_ = Variable(img_)                     # Convert to Variable
    return img_

model = Net("darknet/cfg/yolov3.cfg")
inp = get_test_input()
pred = model(inp, torch.cuda.is_available())
print (pred)