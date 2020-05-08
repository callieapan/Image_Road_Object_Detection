# %load run_ben_net_down.py
import subprocess
import os
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


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from data_helper import UnlabeledDataset, LabeledDataset
#from data_helper_triangle_down import TriangleLabeledDataset,
#from shape_splitter import get_mask_name
#from data_helper_triangle_down import load_mask
from helper import collate_fn, draw_box
import argparse
import datetime


def get_mask_name(camera,shape):
    return camera.replace(".jpeg",f"_{shape[0]}.npy")



def load_mask(camera,downsample_shape):
    mask_name = get_mask_name(camera,downsample_shape)
    if os.path.exists(mask_name):
        mask = np.load(mask_name)
    else:
        save_masks(downsample_shape)
        mask = np.load(mask_name)
    mask = mask.reshape(downsample_shape).transpose()
    return mask



def timeStamped(fname, fmt='%Y-%m-%d-%H-%M-%S_{fname}'):
    return datetime.datetime.now().strftime(fmt).format(fname=fname)


class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

def gen_train_val_index(labeled_scene_index):
    breakpt = len(labeled_scene_index)//3
    labeled_scene_index_shuf = labeled_scene_index
    random.shuffle(labeled_scene_index_shuf)

    train_labeled_scene_index = labeled_scene_index_shuf[:-breakpt]
    val_labeled_scene_index = labeled_scene_index_shuf[-breakpt: ]
    return train_labeled_scene_index, val_labeled_scene_index

def save_file_to_cloud(filename):
    current_dir = os.getcwd()
    path = os.path.join(current_dir,"bucket_upload.sh")
    result = subprocess.run(f"{path} {filename}",shell=True)
    return result

def download_file(filename,cloud_filename):
    current_dir = os.getcwd()
    path = os.path.join(current_dir,"download_bucket.sh")
    command = f"{path} {cloud_filename} {filename}"
    print(f"running {command}")
    if cloud_filename:
        subprocess.run(command,shell=True)
    print(f"loading {filename}")
    return filename

def save_torch(filename,item,cloud=True):
    res = torch.save(item, filename)
    if cloud:
        res = save_file_to_cloud(filename)
    return res

def save_cam_model(cam,item,cloud = True):
    filename = "./models/resnet_1"+cam.replace(".jpeg",".pt")
    save_torch(filename,item,cloud)
    fe_filename = "latest_fe_sd.pt"
    filename = f"./models/{fe_filename}"
    return save_torch(filename,item['feat_extractor_state_dict'],cloud)
    


        
    
    
def load_object(filename,cloud_filename=None):
    if cloud_filename:
        print(f"Downloading {cloud_filename}")
        download_file(filename,cloud_filename)
    print(f"Loading {filename}")
    return torch.load(filename)




def get_output_layer_size(cam,downsample_shape):
    return load_mask(get_mask_name(cam,downsample_shape),downsample_shape).sum()

def get_output_layer(input_size,output_layer_size):
    return nn.Linear(input_size,output_layer_size)

def create_blank_model(base_model,last_layer_size,output_layer_size):
    base_model = base_model()
    base_model.fc = Identity()
    output_layer = get_output_layer(last_layer_size,output_layer_size)
    return nn.Sequential(base_model,output_layer,nn.Sigmoid())

def get_feature_extractor(base_model):
    model = base_model()
    model.fc = Identity()
    return model

def load_model_with_state_dicts(base_model,feature_extractor_sd,output_layer_sd):  
    fe = get_feature_extractor(base_model)
    fe.load_state_dict(feature_extractor_sd)
    
    output_layer = get_output_layer(output_layer_sd["weight"][0].size()[0],\
                                    output_layer_size=output_layer_sd["bias"].size()[0])
    output_layer.load_state_dict(output_layer_sd)
    return nn.Sequential(fe,output_layer,nn.Sigmoid())


def create_model_with_feature_extractor(feature_extractor,last_layer_size,output_layer_size):
    feature_extractor.fc = Identity()
    output_layer = get_output_layer(last_layer_size,output_layer_size)
    return nn.Sequential(feature_extractor,output_layer,nn.Sigmoid())


# def load_cam_model(cam,latest_fe = True,cloud = True,requires_grad=True):
#     cam_short = cam.replace('jpeg','pt')
#     cloud_filename = f"resnet_1{cam_short}"
#     filename = f"./models/{cloud_filename}"
#     if cloud:
#         out_sd = load_object(filename,cloud_filename)
#     else:
#         out_sd = load_object(filename)
#     out_sdr = out_sd["output_layer_state_dict"]
#     if latest_fe:
#         cloud_filename = "latest_fe_sd.pt"
#         filename = f"./models/{cloud_filename}"
#         if cloud:
#             fe_sd = load_object(filename,cloud_filename)
#         else:
#             fe_sd = load_object(filename)
#     else:
#         fe_sd = out_sd["feat_extractor_state_dict"]
    
#     model = load_model_with_state_dicts(models.resnet18,fe_sd,out_sdr)
#     if not requires_grad:
#         for param in model.parameters():
#             param.requires_grad=False
#     return model


def load_cam_model(cam, checkpoint, latest_fe=True, cloud = False,requires_grad=True):
    #cloud is always False
    cam_short = cam.replace('jpeg','pt')
    out_sdr = checkpoint[f"resnet_1{cam_short}"]["output_layer_state_dict"]
    #latest_fe is always true
    fe_sd = checkpoint["latest_fe_sd.pt"]
    
    model = load_model_with_state_dicts(models.resnet18,fe_sd,out_sdr)
    if not requires_grad:
        for param in model.parameters():
            param.requires_grad=False
    return model
    
    

def degrad_layers(model,layers):
    for layer in layers:
        for param in model[layer].parameters():
            param.requires_grad = False
            
def grad_layers(model,layers):
    for layer in layers:
        for param in model[layer].parameters():
            param.requires_grad = True

def test_model(loader, model):
    """
    Help function that tests the model's performance on a dataset
    @param: loader - data loader for the dataset to test against
    """
     
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for i ,(sample, target, road_image, extra, road_image_mod) in enumerate(loader):
             
            sample_ = torch.stack(sample,0).cuda() #should be [batch size,3, h,w]
            
            labels = torch.stack(road_image_mod, 0).cuda()
            
            
            outputs = model(sample_)
            predicted = (outputs>0.5).int() ## convert to bineary
            
            total += (labels.size(0)*labels.size(1))
            correct += predicted.eq(labels.int()).sum().item()
        
    return (100 * correct / total)




def train(feat_extractor, **train_kwargs):
    #save model
    if not os.path.exists("./models"):
        os.mkdir("models")
    
    print("args to train function:")
    print(train_kwargs)
    
    for cycle in range(train_kwargs["train_cycles"]):
        for cam in image_names: #let's try just front camera
            print("training {}".format(cam))
            #make camera specific train loader
            labeled_trainset = training_tools[cam][1]
            train_loader = torch.utils.data.DataLoader(labeled_trainset , batch_size=train_kwargs["batch"], 
                                                      shuffle=True, num_workers=2, collate_fn=collate_fn)
            labeled_valset = training_tools[cam][2]
            val_loader = torch.utils.data.DataLoader(labeled_valset , batch_size=train_kwargs["batch"], 
                                                      shuffle=True, num_workers=2, collate_fn=collate_fn)


            if train_kwargs.get("load_models",False): 
                model = load_cam_model(cam,latest_fe=True,cloud=train_kwargs["load_cloud"]).cuda()
                
            else:
                output_layer = training_tools[cam][0] #output the layer


                #make camera spcific model
                model = nn.Sequential(feat_extractor, output_layer, nn.Sigmoid()).cuda()


            criterion = torch.nn.BCELoss(reduction = 'sum') #trying summation
            train_losses = []
            val_accs = []

            for e in range(train_kwargs["epochs"]):
                print(f"epoch: {e}")
                t = time.process_time()
                if e < train_kwargs["eto"]:
                    print("training output layer")
                    degrad_layers(model,[0]) #degrad the base model
                else:
                    print("training whole network")
                    grad_layers(model,[0])

                param_list = [p for p in model.parameters() if p.requires_grad]
                optimizer = torch.optim.Adam(param_list, lr=train_kwargs["lr"], eps=train_kwargs["eps"])

                for i ,(sample, target, road_image, extra, road_image_mod) in enumerate(train_loader):

                    sample_ = torch.stack(sample,0).cuda() #should be [batch size,3, h,w]
                    labels = torch.stack(road_image_mod, 0).cuda() #should be [batch size, cropsize]

                    optimizer.zero_grad()
                    outputs = model(sample_) 

                    loss = criterion(outputs, labels.float())
                    loss.backward()
                    optimizer.step()
                    train_losses.append(loss.item())

                # validate every 200 iterations
                val_acc = test_model(val_loader, model) #calls model.eval()
                val_accs.append(val_acc)
                #do some stuff
                elapsed_time = time.process_time() - t
                print('Epoch: [{}], Step: [{}], Train Loss {:.4f}, Validation Acc: {:.4f}, time {:.4f}'.format( 
                           e+1, i+1, loss,  val_acc, elapsed_time))
                model.train() #go back to training
                t = time.process_time()

            print("save camera model") 
            item = {

                'model_state_dict': model.state_dict(),
                'feat_extractor_state_dict':  feat_extractor.state_dict(),
                'output_layer_state_dict': model[1].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_accs': val_accs
                }
            save_cam_model(cam,item,cloud=train_kwargs["save_cloud"])
            if cycle == (train_kwargs["train_cycles"] -1 ) and\
            train_kwargs["last_save"] and\
            not train_kwargs["save_cloud"]:
                save_cam_model(cam,item,cloud=True)

    

    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Run neural net, first argument is downsampling rate')
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", help="The downsample size for the image (1 dimension)",default=100,
                        type=int)
    parser.add_argument("--b", help="batch-size",default=100,
                        type=int)
    parser.add_argument("--e", help="epochs",default=5,
                    type=int)
    parser.add_argument("--s", help="save to cloud? default no",default=0,
                    type=int)
    parser.add_argument("--l", help="load from cloud? default no",default=0,
                    type=int)    
    parser.add_argument("--tc", help="training_cycles",default=1,
                    type=int)    
    parser.add_argument("--ls",help='save last cycle',default=1,type=int)
    parser.add_argument("--eto",help='how many epochs to train output layer alone',default=0,type=int)
    args = parser.parse_args()
    downsample_shape = (args.d,args.d)
    
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0);


    image_folder = 'data'
    annotation_csv = 'data/annotation.csv'

    
    unlabeled_scene_index = np.arange(106)
    labeled_scene_index = np.arange(106, 134)
    
    
    normalize = torchvision.transforms.Normalize(mean=[0.6394939, 0.6755114, 0.7049375],
                                         std=[0.31936955, 0.3117349 , 0.2953726 ])

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                               normalize
                                               ])

    train_labeled_scene_index, val_labeled_scene_index = gen_train_val_index(labeled_scene_index)
    crop_size = {cam:get_output_layer_size(cam,downsample_shape) for cam in image_names}
    print(crop_size)
    training_tools = {cam: (nn.Linear(512, crop_size[cam]), 
                           #training set
                           TriangleLabeledDataset(image_folder=image_folder,
                                  annotation_file=annotation_csv,
                                  scene_index=train_labeled_scene_index,
                                  transform=transform,
                                  extra_info=True,
                                camera = cam,downsample_shape=downsample_shape),
                           #validation set
                            TriangleLabeledDataset(image_folder=image_folder,
                                  annotation_file=annotation_csv,
                                  scene_index=val_labeled_scene_index,
                                  transform=transform,
                                  extra_info=True,
                                camera = cam,downsample_shape=downsample_shape),
                       
                       
                       ) for cam in image_names}
    
    feat_extractor = torchvision.models.resnet18()
    feat_extractor.fc = Identity() #change it to identity

    train_kwargs={
        'epochs':args.e,
        'lr': 2e-05,
        'momentum': 0.99,
        'eps':1e-08,
        'batch':args.b,
        'load_models':True,
        'load_cloud': args.l,
        'save_cloud':args.s,
        'train_cycles':args.tc,
        'lsat_save':args.ls,
        'eto':args.eto
        }
    
    
    train(feat_extractor, **train_kwargs)
    
    print('finished')

#sample command
#python test_nn.py --d 100 --b 100 --e 5 --s 0 --l 0 --tc 10 --ls 1 --eto 1
