import torch
import torch.nn as nn

import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

import numpy as np
import math
import sys
import time
import random

import torchvision.models.detection.mask_rcnn
 
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.rpn import RPNHead

def fr50_Model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    # create an anchor_generator for the FPN
    # which by default has 5 outputs
    
    anchor_generator = AnchorGenerator(
        #sizes=tuple([(16, 32, 64, 128, 256, 512) for _ in range(5)]),
        sizes=tuple([(10, 15, 20, 30, 40) for _ in range(5)]),
         
        aspect_ratios=tuple([(0.25, 0.5, 1.0, 2.0) for _ in range(5)]))
    
    
    
    model.rpn.anchor_generator = anchor_generator

    # 256 because that's the number of features that FPN returns
    model.rpn.head = RPNHead(256, anchor_generator.num_anchors_per_location()[0])
    return model

def sew_images(sing_samp):
        # sing_samp is [6, 3, 256, 306], one item is batch
        # output is the image object of all 6 pictures 'sown' together
        #############
        # A | B | C #
        # D | E | F #
        #############
        
        # return [3, 768, 612]
        
        A1 = sing_samp[0][0]
        A2 = sing_samp[0][1]
        A3 = sing_samp[0][2]

        B1 = sing_samp[1][0]
        B2 = sing_samp[1][1]
        B3 = sing_samp[1][2]

        C1 = sing_samp[2][0]
        C2 = sing_samp[2][1]
        C3 = sing_samp[2][1]

        D1 = sing_samp[3][0]
        D2 = sing_samp[3][1]
        D3 = sing_samp[3][2]

        E1 = sing_samp[4][0]
        E2 = sing_samp[4][1]
        E3 = sing_samp[4][2]

        F1 = sing_samp[5][0]
        F2 = sing_samp[5][1]
        F3 = sing_samp[5][2]

        #print("F shape {}".format(F1.shape))

        T1 = torch.cat([A1, B1, C1], 1)
        T2 = torch.cat([A2, B2, C2], 1)
        T3 = torch.cat([A3, B3, C3], 1)

        B1 = torch.cat([D1, E1, F1], 1)
        B2 = torch.cat([D2, E2, F2], 1)
        B3 = torch.cat([D3, E3, F3], 1)
        #print("T1 shape {}".format(T1.shape))

        comb1 = torch.cat([T1,B1], 0)
        comb2 = torch.cat([T2,B2], 0)
        comb3 = torch.cat([T3,B3], 0)

        #print("comb1 shape {}".format(comb1.shape)) #should be 768, 612
        comb = torch.stack([comb1, comb2, comb3])
        toImg = transforms.ToPILImage()
        result = toImg(comb) # image object [3, 768, 612]
        return result

def sew_images_panorm(samples , to_img = False):
    new_samples = []
    for sample in samples:
        new_sample =  torch.cat([sample[0],sample[1],sample[2],sample[5],sample[4],sample[3]],axis = 2)
        toImg = transforms.ToPILImage()
        if to_img:
            new_sample = toImg(new_sample)
        new_samples.append(new_sample)
    return new_samples #list of [3, 256, 1836]


def convert_boxes(sing_pred, device):
    #convert predicted boxes in 800 x 800 to reg. convenction
    #pred box is for a single batch
    box = sing_pred["boxes"] #tensor of N by 4
    box = (box -400)/10
    N = box.shape[0]
    result = torch.zeros((N, 8)).to(device)
    #box
    #[xmin, ymin, xmax, ymax]
    
    #result
    #['fl_x', 'fr_x', 'bl_x', 'br_x', 'fl_y', 'fr_y','bl_y', 'br_y']
    #[xmax,    xmax,   xmin,   xmin,   ymax,   ymin,  ymax,  ymin]
    
    result[:, 0] = box[:,2]
    result[:, 1] = box[:,2]
    result[:, 2] = box[:,0]
    result[:, 3] = box[:,0]
    result[:, 4] = box[:,3]
    result[:, 5] = box[:,1]
    result[:, 6] = box[:,3]
    result[:, 7] = box[:,1]
    
    
    result = result.view(N, 2, 4)
    return result
