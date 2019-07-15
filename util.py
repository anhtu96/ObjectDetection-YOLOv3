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

def process_prediction(x, dims, anchors, num_classes):
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
    
    # Scale back to normal size for bx, by, bw, bh
    x[:,:,:4] *= scale_h
    return x

def unique_class(classes):
    return torch.unique(classes, dim=-1)

def write_results(prediction, thresh_pred=0.4, iou_thresh=0.5):
    batch_size = prediction.size(0)
    write = False
    
    conf_mask = (prediction[:,:,4] > thresh_pred).float().unsqueeze(2)
    pred_mask = conf_mask * prediction
    
    pred_box_corners = pred_mask.copy_(pred_mask)
    pred_box_corners[:,:,0] = pred_mask[:,:,0] - pred_mask[:,:,2]/2
    pred_box_corners[:,:,1] = pred_mask[:,:,1] - pred_mask[:,:,3]/2
    pred_box_corners[:,:,2] = pred_mask[:,:,0] + pred_mask[:,:,2]/2
    pred_box_corners[:,:,3] = pred_mask[:,:,1] + pred_mask[:,:,3]/2
    
    for i in range(batch_size):
        img_pred = pred_box_corners[i]
        scores, classes = torch.max(img_pred[:,5:], dim=-1, keepdim=True)
        img_pred = torch.cat((img_pred[:,:5], scores.float(), classes.float()), dim=1)
        nonzero_idx = torch.nonzero(img_pred[:,4]).squeeze(1)
        if (nonzero_idx.size(0) > 0):
            img_pred_ = img_pred[nonzero_idx]
        img_classes = unique_class(img_pred_[:,-1])
        for cl in img_classes:
            cl_mask = img_pred_ * (img_pred_[:,-1] == cl).float().unsqueeze(1)
            cl_mask_nonzero = torch.nonzero(cl_mask[:,-2]).squeeze()
            img_pred_class = img_pred_[cl_mask_nonzero].view(-1,7)
            conf_sort_val, conf_sort_idx = torch.sort(img_pred_class[:,4], descending=True)
            img_pred_class = img_pred_class[conf_sort_idx].view(-1, 7)
            len_img_pred = img_pred_class.size(0)
            
            for idx in range(len_img_pred):
                iou = calc_iou(img_pred_class[i], img_pred_class[(idx+1):])
                iou_mask = (iou < iou_thresh).float().unsqueeze(1)
                img_pred_class[idx+1:] *= iou_mask
                nonzero_idx = torch.nonzero(img_pred_class[:,4]).squeeze()
                img_pred_class = img_pred_class[nonzero_idx].view(-1,7)
            batch_ind = img_pred_class.new(img_pred_class.size(0), 1).fill_(i)      #Repeat the batch_id for as many detections of the class cls in the image
            seq = batch_ind, img_pred_class
            if not write:
                output = torch.cat(seq,1)
                write = True
            else:
                out = torch.cat(seq,1)
                output = torch.cat((output,out))
        return output
                
                
def calc_iou(box1, boxes2):
    xi1 = torch.max(box1[0], boxes2[:,0])
    yi1 = torch.max(box1[1], boxes2[:,1])
    xi2 = torch.min(box1[2], boxes2[:,2])
    yi2 = torch.min(box1[3], boxes2[:,3])
    intersection = torch.clamp(xi2-xi1, min=0) * torch.clamp(yi2-yi1, min=0)
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_boxes2 = (boxes2[:,2] - boxes2[:,0]) * (boxes2[:,3] - boxes2[:,1])
    union = area_box1 + area_boxes2 - intersection
    iou = intersection / union
    return iou