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

# from coco_utils import get_coco_api_from_dataset
# from coco_eval import CocoEvaluator
#import utils



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
    box = (box -400)/100
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
