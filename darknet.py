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
            padding = int(layer['pad'])
            
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, padding)
            module.add_module('conv_{}'.format(idx), conv)
            if batchnorm:
                bn = nn.BatchNorm2d(filters)
                module.add_module('batch_norm_{}'.format(idx), bn)
                
            act_layer = None
            if activation == 'leaky':
                act_layer = nn.LeakyReLU(0.1) # 0.1 according to YOLOv1 paper
                
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
#            if end_layer == 0:
#                filters = output_filters[start_layer + idx]
#            else:
#                filters = output_filters[start_layer + idx] + output_filters[end_layer]
                
        # Shortcut layer (skip connection)
        elif layer['type'] == 'shortcut':
            shortcut = EmptyLayer()
            module.add_module('shortcut_{}'.format(idx), shortcut)
            
        # YOLO layer
        elif layer['type'] == 'yolo':
            mask = [int(x) for x in layer['mask'].split(',')]
            anchors = [int(x) for x in layer['anchors'].split(',')]
            anchors = [(anchors[2*i], anchors[2*i+1]) for i in mask]
            yolo = YoloLayer()
            module.add_module('yolo_{}'.format(idx), yolo)