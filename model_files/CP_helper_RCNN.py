
import torch
import torch.nn as nn
#import torch.nn.functional as F
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

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
import utils


normalize = transforms.Normalize(mean=[0.6394939, 0.6755114, 0.7049375],
                                     std=[0.31936955, 0.3117349 , 0.2953726 ])


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



def gen_train_val_index(labeled_scene_index):
    breakpt = len(labeled_scene_index)//3
    labeled_scene_index_shuf = labeled_scene_index
    random.shuffle(labeled_scene_index_shuf)

    train_labeled_scene_index = labeled_scene_index_shuf[:-breakpt]
    val_labeled_scene_index = labeled_scene_index_shuf[-breakpt: ]
    return train_labeled_scene_index, val_labeled_scene_index    
    
    
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


    
    
class Encoder(nn.Module):

    
    def __init__(self):
        super(Encoder, self).__init__()
        
        self.encoder = models.resnet18() #[6, 3, w, h]
        self.encoder.fc = Identity() #set last layer to identity, output is [6, 512]
      
        
    def forward(self, x): #x should be [batch, 1, 256, 306] for a single image
        output = self.encoder(x)#should be [6, 512]
        return output  

class UpModel(nn.Module):
    
    def testfunc():
        print("hey")
    
    def forward(self, x):
        #print("in forward")
        output = self.main(x) 
        return output
        
    def __init__(self, in_channel, out_channel):
        super(UpModel, self).__init__()
        #self.ngpu = ngpu
        ngf = 64
        
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( in_channel, ngf * 48, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 48),
            nn.ReLU(True),
            # state size. (ngf*48) x 4 x 4
            nn.ConvTranspose2d(ngf * 48, ngf * 24, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 24),
            nn.ReLU(True),
            # state size. (ngf*24) x 8 x 8
            nn.ConvTranspose2d( ngf * 24, ngf * 12, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 12),
            nn.ReLU(True),
            # state size. (ngf*12) x 16 x 16
            nn.ConvTranspose2d( ngf * 12, ngf * 6, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 6),
            nn.ReLU(True),
            # state size. (ngf * 6) x 32 x 32
            nn.ConvTranspose2d( ngf *6 , ngf * 3,  4, 2, 1, bias=False),
            #nn.Tanh()
            nn.BatchNorm2d(ngf * 3),
            nn.ReLU(True),
            # state size. (ngf * 3) x 64 x 64
    
            #from here on just scale up
            nn.ConvTranspose2d( ngf*3, ngf,  5, 3, 0, bias=False), 
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
    
            nn.ConvTranspose2d( ngf, ngf,  5, 2, 0, bias=False), 
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
        
            nn.ConvTranspose2d( ngf, ngf,  5, 2, 0, bias=False), 
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            *([nn.ConvTranspose2d( ngf, ngf,  5, 1, 0, bias=False), 
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)]*3),
            
            #nn.ConvTranspose2d( ngf, ngf,  5, 1, 1, bias=False), #add padding to fit into 800 
            nn.ConvTranspose2d( ngf, out_channel,  6, 1, 1, bias=False),
            nn.Tanh()
)
        
        
        
class CombModel (nn.Module):
      
    def get_instance_segmentation_model(self, num_classes, pretrain = False):
        # load an instance segmentation model , if pretrain = True, it is pre-trained on COCO
        if pretrain:
            #print("in pretrain")
            model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True) #try with pretrained first
        else: 
            model = torchvision.models.detection.maskrcnn_resnet50_fpn()
        # get the number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # now get the number of input features for the mask classifier
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # and replace the mask predictor with a new one
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                           hidden_layer,
                                                           num_classes)
        return model
    
    def __init__(self):
        super(CombModel, self).__init__()
        self.encoder = Encoder()
        self.decoder = UpModel(3072, 1)
        self.maskRCNN = self.get_instance_segmentation_model(num_classes = 2)
        
    def forward(self, image, target = None): 
        #image is tuple([6, 3, 256, 306]), length 1, target is the dictionary of stuff
        #target is a tuple if ( dict of boxes, masks etc) length 1
        six_encode = self.encoder(image[0]) # output [6, 512]
        six_encode = six_encode.view(3072,1,1).unsqueeze(0) #[1, 3072, 1, 1]
        #print("six_encode_shape {}".format(six_encode.shape))
        dec_output = self.decoder(six_encode) #[1, 1, 800, 800]
        #print("decode_output_shape {}".format(dec_output.shape))
           
        #output_dict = self.maskRCNN(dec_output)
        dec_output = tuple([dec_output.squeeze(0)]) #turn it into a tuple of [1, 800, 800] , length 1  
        loss_dict = None
        pred = None
        
        if target is not None:
            loss_dict = self.maskRCNN(dec_output, target)           
        else:
            pred = self.maskRCNN(dec_output)
        
        return pred, loss_dict
   
    
def trans_target(old_targets): #target from the given dataset and data loader
    tg_list = []
    for tg_ in old_targets: #for each item in the batch
        target = {}
        corners = tg_['bounding_box'].view(-1, 1, 8).squeeze(1).numpy()
        boxes = get_boxes(corners)
        
        categories = tg_['category'].numpy() #switch to array
        labels = convert_categories(categories)
        masks = gen_masks( corners , labels) 
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) #this may need to be rounded but leave for now
        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(labels)), dtype=torch.int64)
        target["boxes"]  = boxes
        target["labels"] = labels 
        target["masks"] = masks
        index = 100 #not sure if this matters
        target["image_id"] = torch.tensor([index])
        target["area"] = area
        target["iscrowd"] = iscrowd

        tg_list.append(target)
        
    return tg_list #removed tuple
              

              
def train_one_epoch_combModel(model, optimizer, data_loader, device, epoch, print_freq): #this data loader is given loader
    
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
#     if epoch == 0:
#         warmup_factor = 1. / 1000
#         warmup_iters = min(1000, len(data_loader) - 1)

#         lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for sample, old_targets, road_image, extra in metric_logger.log_every(data_loader, print_freq, header): 
        
        images = sample
        targets = trans_target(old_targets)
        #print("images len {}, targets len {}".format(len(images), len(targets)))
        #print("images[0] shape {}".format(images[0].shape)) # [6, 3, 256, 306]      
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        _ , loss_dict = model(images, targets)
        #print(loss_dict)
        
        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def train_one_epoch_FastRCNN(model, optimizer, data_loader, device, epoch, print_freq, mode = "sew6", encoder = None,train_encoder= False ): 
    #this data loader is given loader
    #mode can be "sew6", "panorm", "autoencode"
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
#     if epoch == 0:
#         warmup_factor = 1. / 1000
#         warmup_iters = min(1000, len(data_loader) - 1)

#         lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
    if mode == 'panorm':
        tt = transforms.Compose([transforms.Resize((800, 800)), transforms.ToTensor(), normalize]) #this is for 6 images combo
    for sample, old_targets, road_image, extra in metric_logger.log_every(data_loader, print_freq, header): 
        
        #images = sample[0] 
        
        targets = trans_target(old_targets)
        #print("images len {}, targets len {}".format(len(images), len(targets)))
        #print("len(sample) {}, sample [0] shape {}".format(len(sample), sample[0].shape)) # [6, 3, 256, 306]      
        #images = list(image.to(device) for image in images)
        if mode == "panorm":
            images = [tt(s).to(device) for s in sew_images_panorm(sample, to_img = True)]
        
        elif mode == "autoencode":
            encoder.cuda()
            samp_pan = sew_images_panorm(sample) #convert to panoramic tensor
            samp_pan = [normalize(i) for i in samp_pan]
            samp_pan_t = torch.stack(samp_pan, dim = 0) #stack
            if train_encoder:
                images = encoder.return_image_tensor(samp_pan_t.to(device),train_encoder) #see if it will take it or it needs to take a list
            else:
                images = encoder.return_image_tensor(samp_pan_t.cuda(),train_encoder).to(device)
             
        
        else: #mode is sew6
            images = [tt(sew_images(s)).to(device) for s in sample] #list of [3, 800, 800], should be 1 per patch
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        #print(loss_dict)
        
        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def get_boxes(corners): #this is the corners of the annotaion file
    # the corners are in meter and time 10 will convert them in pixels
    # Add 400, since the center of the image is at pixel (400, 400)
    # The negative sign is because the y axis is reversed for matplotlib
    #ax.plot(point_squence.T[0] * 10 + 400, -point_squence.T[1] * 10 + 400, color=color)
    
    
    #['fl_x', 'fr_x', 'bl_x', 'br_x', 'fl_y', 'fr_y','bl_y', 'br_y']
    #translate this to boxes to the fastRNN format
    xvals = corners[:, :4] *10 +400
    #yvals = -(corners[:, 4:]*10 +400) #not flipping the y vals
    yvals = (corners[:, 4:]*10 +400)
    boxes = []
    num_obj = corners.shape[0]
    #print(corners.shape, num_obj)
    for i in range (num_obj):
        xmin = np.min(xvals[i])
        xmax = np.max(xvals[i])
        ymin = np.min(yvals[i])
        ymax = np.max(yvals[i])
    
        boxes.append([xmin, ymin, xmax, ymax])

    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    
    return boxes
def convert_categories(categories):
    #Old categories
     
    # 'other_vehicle': 0,
    # 'bicycle': 1,
    # 'car': 2,
    # 'pedestrian': 3,
    # 'truck': 4,
    # 'bus': 5,
    # 'motorcycle': 6,
    # 'emergency_vehicle': 7, 
    # 'animal': 8
    
    
    #New categories
     
    # 'car': 1,
    # 'pedestrian': 2,
    # 'all other': 3,
     
    map_dict = {0:1, 1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1}
    labels = []
    for c in categories:
        labels.append(map_dict[c])
    return torch.tensor(labels)

def gen_masks(corners , labels, img_w = 800, img_h = 800):
    '''
    essentially fill in the boxes in road_image with the class labels
    however all background is 0, hence no road is shown
    corners format: ['fl_x', 'fr_x', 'bl_x', 'br_x', 'fl_y', 'fr_y','bl_y', 'br_y']
    '''
    #print('corners shape {}'.format(corners.shape))
    corners = corners*10 +400 #convert into the road image format of 800, 800 with center being 400, 400
    xvals = np.round(corners[:, :4], 0).astype(int)
    yvals = -np.round(corners[:, 4:], 0).astype(int) #keep this negative for the chart
    num_obj = len(labels)
    #print('num_obj {}'.format(num_obj))
    masks = torch.zeros((num_obj, img_w, img_h))
    
    for i in range(num_obj):
        colmin = np.min(xvals[i])
        colmax = np.max(xvals[i])
        
        rowmin = np.min(yvals[i])
        rowmax = np.max(yvals[i])
        #print("mask shape {}".format(masks.shape))
        #print("i {}, xmin {}, xmax {}, ymin {}, ymax {} label {}".format(i, xmin, xmax, ymin, ymax, labels[i]))
        masks[i, rowmin:rowmax, colmin:colmax] = labels[i]
       
    return masks            
              
              
def gen_result_chart(sing_target, img_w = 800, img_h = 800):
    '''
    essentially fill in the boxes in road_image with the class labels
    however all background is 0, hence no road is shown
    boxes format: ['col min', 'row_min', 'col_max', 'row_max']
    '''
 
    boxes = sing_target["boxes"].cpu().numpy()
    labels = sing_target["labels"].cpu().numpy()
    num_obj = len(labels)
    #print('num_obj {}'.format(num_obj))
    background = torch.zeros((img_w, img_h))
    
    for i in range(num_obj):
        colmin = int(round(boxes[i][0]))
        colmax = int(round(boxes[i][2]))
        
        rowmin = int(round(boxes[i][1]))
        rowmax = int(round(boxes[i][3]))
        #print(rowmin, rowmax, colmin, colmax)
        background[rowmin:rowmax, colmin:colmax] = int(labels[i]) #flip y axis to stay with class convention
       
    return background            
              
              
              
              
              
              
              
              