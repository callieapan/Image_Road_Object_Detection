import os
import sys
import random
from PIL import Image
import numpy as np
import pandas as pd
#for image transform
import cv2

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.figsize'] = [5, 5]
matplotlib.rcParams['figure.dpi'] = 200

import random
import time

#sys.path.append('/root/dl2020')
from data_helper import UnlabeledDataset, LabeledDataset
from helper import collate_fn, draw_box
#from Unet import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.utils.data



def gen_train_val_index(labeled_scene_index):
    breakpt = len(labeled_scene_index)//3
    labeled_scene_index_shuf = labeled_scene_index
    random.shuffle(labeled_scene_index_shuf)

    train_labeled_scene_index = labeled_scene_index_shuf[:-breakpt]
    val_labeled_scene_index = labeled_scene_index_shuf[-breakpt: ]
    return train_labeled_scene_index, val_labeled_scene_index



# random.seed(0)
# np.random.seed(0)
# torch.manual_seed(0);
# image_folder = '../data'
# annotation_csv = '../data/annotation.csv'

# unlabeled_scene_index = np.arange(106)
# # The scenes from 106 - 133 are labeled
# # You should devide the labeled_scene_index into two subsets (training and validation)
# labeled_scene_index = np.arange(106, 134)
# train_labeled_scene_index, val_labeled_scene_index  = gen_train_val_index(labeled_scene_index)


# normalize = torchvision.transforms.Normalize(mean=[0.6394939, 0.6755114, 0.7049375],
#                                      std=[0.31936955, 0.3117349 , 0.2953726 ])

# transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
#                                            normalize
#                                            ])


# kwargs = {
#     #'first_dim': 'sample',
#     'transform': transform,
#     'image_folder': image_folder,
#     'annotation_file': annotation_csv,
#     'extra_info': True}

# #dataset_train = LabeledDataset_RCNN (scene_index=train_labeled_scene_index, **kwargs)
# #dataset_val = LabeledDataset_RCNN (scene_index=val_labeled_scene_index, **kwargs)

# dataset_train = LabeledDataset(scene_index=train_labeled_scene_index, **kwargs)
# dataset_val = LabeledDataset(scene_index=val_labeled_scene_index, **kwargs)





# train_data_loader = torch.utils.data.DataLoader(
#     dataset_train, batch_size=30, shuffle=False, num_workers=4,
#     collate_fn=collate_fn)

# val_data_loader = torch.utils.data.DataLoader(
#     dataset_val, batch_size=30, shuffle=False, num_workers=4,
#     collate_fn=collate_fn)




#BEN CHANGED
# plot_image(sample[0].reshape(1,3,256,-1))
def sew_images(samples):
    new_samples = []
    for sample in samples:
        new_sample =  torch.cat([sample[0],sample[1],sample[2],sample[5],sample[4],sample[3]],axis = 2)
        new_samples.append(new_sample)
    return new_samples

def plot_image(img):
    plt.imshow(img.numpy().transpose(1, 2, 0))
    plt.axis('off');
    plt.show()
    #fig, ax = plt.subplots()

# define the NN architecture
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        ## encoder layers ##
        # conv layer (depth from 1 --> 16), 3x3 kernels
        #self.conv1 = nn.Conv2d(3, 16, 3, padding=1)  
        # conv layer (depth from 16 --> 8), 3x3 kernels
        #self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        #self.pool = nn.MaxPool2d(2, 2)
        
        ## decoder layers ##
        #self.conv4 = nn.Conv2d(4, 16, 3, padding=1)
        #self.conv5 = nn.Conv2d(16, 1, 3, padding=1)
        
#         #V1 Kind of works
#         self.enc_conv1 = nn.Conv2d(3, 64, 3, padding=(1,1),stride=(1,5))
#         self.enc_conv2 = nn.Conv2d(64, 128, 3, padding=(1,8),stride=(2,3))
#         self.z = nn.ConvTranspose2d(128,3,3,stride=7,padding=46)
#         self.dec_conv1 = nn.Conv2d(3,128,3,stride=7,padding=46)
#         self.dec_conv2 = nn.ConvTranspose2d(128,64,3,stride=(2,3),padding=(1,8),output_padding=(1,0))
#         self.dec_conv3 = nn.ConvTranspose2d(64,3,3,stride=(1,5),padding=(1,1))

        self.enc_conv1 = nn.Conv2d(3, 16, 3, padding=(1,17),stride=(1,2))
        self.enc_conv2 = nn.Conv2d(16, 32, 3, padding=(1,17),stride=(1,2))
        self.enc_conv3 = nn.Conv2d(32, 64, 3, padding=(1,15),stride=(1,2))
        #pool = nn.MaxPool2d(2,2)
        self.z = nn.ConvTranspose2d(64,3,3,stride=3)
        
        self.dec_conv1 = nn.Conv2d(3,64,3,stride=3,padding=0)
        #dec_inv_pool = upsample
        self.dec_conv2 = nn.ConvTranspose2d(64, 32, 3, padding=(1,15),stride=(1,2))
        self.dec_conv3 = nn.ConvTranspose2d(32, 16, 3, padding=(1,17),stride=(1,2),output_padding=(0,1))
        self.dec_conv4 = nn.ConvTranspose2d(16, 3, 3, padding=(1,17),stride=(1,2),output_padding=(0,1))
        
        #for encoding
#         self.enc_conv1 = nn.Conv2d(3, 16, 3, padding=(1,17),stride=(1,2))
#         self.enc_conv2 = nn.Conv2d(16, 32, 3, padding=(1,17),stride=(1,2))
#         self.enc_conv3 = nn.Conv2d(32, 64, 3, padding=(1,15),stride=(1,2))
#         self.pool = nn.MaxPool2d(2,2)
#         self.z = nn.ConvTranspose2d(64,3,3,stride=7,padding=46)
        #for decoding
#         self.dec_conv1 = nn.Conv2d(3,64,3,stride=7,padding=46)
#         self.dec_inv_pool = self.upsample
#         self.dec_conv2 = nn.ConvTranspose2d(64, 32, 3, padding=(1,15),stride=(1,2))
#         self.dec_conv3 = nn.ConvTranspose2d(32, 16, 3, padding=(1,17),stride=(1,2),output_padding=(0,1))
#         self.dec_conv4 = nn.ConvTranspose2d(16, 3, 3, padding=(1,17),stride=(1,2),output_padding=(0,1))
    def return_image_tensor(self,x,requires_grad = False):
        if requires_grad:
            a = self.encode(x)
        else:
            with torch.no_grad():
                a = self.encode(x)
        return F.pad(a,pad=(16,16,16,16))
    def encode(self,x):
        #V1 encoder
        x = torch.tanh(self.enc_conv1(x))
        x = torch.tanh(self.enc_conv2(x))
        x = torch.tanh(self.enc_conv3(x))
        x = torch.sigmoid(self.z(x))
#         x = F.relu(self.enc_conv1(x))
#         x = F.relu(self.enc_conv2(x))
#         x = F.relu(self.enc_conv3(x))
#         x = self.pool(x)
#         x = F.relu(self.z(x))
        return x
    
    def decode(self,x):
        ## V1 decoder 
        x = torch.tanh(self.dec_conv1(x))
        x = torch.tanh(self.dec_conv2(x))
        x = torch.tanh(self.dec_conv3(x))
        x = torch.tanh(self.dec_conv4(x))
        
#         x = F.relu(self.dec_conv1(x))
#         x = self.dec_inv_pool(x)
#         x = F.relu(self.dec_conv2(x))
#         x = F.relu(self.dec_conv3(x))
#         x = F.relu(self.dec_conv4(x))
        return x
    
    def forward(self, x):
        # upsample, followed by a conv layer, with relu activation function  
        # this function is called `interpolate` in some PyTorch versions
        #x = F.upsample(x, scale_factor=2, mode='nearest')
        #x = F.relu(self.conv4(x))
        # upsample again, output should have a sigmoid applied
        #x = F.upsample(x, scale_factor=2, mode='nearest')
        #x = F.sigmoid(self.conv5(x))
        ##x = self.pool(x)  # compressed representation
        
        x = self.encode(x)
        x = self.decode(x)
        
        
        return x
    
def get_autoencoder( checkpoint, require_grad = False):
    m_test = ConvAutoencoder()
    #m_test.load_state_dict(torch.load('autoencoder_new.pt2'))
    m_test.load_state_dict(checkpoint['autoencoder_new.pt2'])
    return m_test