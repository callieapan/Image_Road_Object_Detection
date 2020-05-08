import torch
import torch.nn as nn
#import torch.nn.functional as F
import torchvision

 
import torchvision.transforms as transforms
from skimage.transform import resize

# import your model class
# import ...
from auto_encoder_submission import *
from bbox import *
from test_nn_submission import *

 
# Put your transform function here, we will use it for our dataloader
# For bounding boxes task
def get_transform_task1(): 
    #load_autoencode
   
    return torchvision.transforms.ToTensor()

# For road map task
def get_transform_task2():
    
    normalize = torchvision.transforms.Normalize(mean=[0.6394939, 0.6755114, 0.7049375],
                                     std=[0.31936955, 0.3117349 , 0.2953726 ])

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                           normalize
                                           ])
    
    return transform





class ModelLoader():
    # Fill the information for your team
    team_name = 'Musketeers'
    team_number = 1
    round_number = 1
    team_member = ["Ben Wolfson", "Calliea Pan", "Arushi Himatsingka"]
    contact_email = 'calliea.pan@nyu.edu'

    def __init__(self, model_file='musketeer.pt'):
        # You should 
        #       1. create the model object
        #       2. load your state_dict
        #       3. call cuda()
        # self.model = ...
        # 
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        checkpoint = load_object(model_file) #torch.load(model_file)
        self.AE = get_autoencoder(checkpoint).to(self.device) #from auto_encoder.py
        bbox_model = fr50_Model() #from bbox.py
        bbox_model.load_state_dict(checkpoint['bbox_state_dict'])
        bbox_model.eval()
        self.bbox_model = bbox_model.to(self.device)
        
        ### load ben's model ###
        
        self.image_names = [
        'CAM_FRONT_LEFT.jpeg',
        'CAM_FRONT.jpeg',
        'CAM_FRONT_RIGHT.jpeg',
        'CAM_BACK_LEFT.jpeg',
        'CAM_BACK.jpeg',
        'CAM_BACK_RIGHT.jpeg'
        ]
        self.models={}
        self.masks={}
        for idx,image in enumerate(self.image_names):
            self.models[image]=load_cam_model(image, checkpoint, latest_fe=True, cloud = False,requires_grad=False).cuda()
            self.masks[image]=load_mask(image,(100,100))
        
        ## end load ben's model##
        self.normalize = transforms.Normalize(mean=[0.6394939, 0.6755114, 0.7049375],
                                     std=[0.31936955, 0.3117349 , 0.2953726 ])
        

    def get_bounding_boxes(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a tuple with size 'batch_size' and each element is a cuda tensor [N, 2, 4]
        # where N is the number of object
        
        samp_pan = sew_images_panorm(samples) #convert to panoramic tensor
        samp_pan = [self.normalize(i) for i in samp_pan]
        samp_pan_t = torch.stack(samp_pan, dim = 0)
        images = self.AE.return_image_tensor(samp_pan_t.to(self.device))
        
        pred = self.bbox_model(images)
        box_list = []
        for p in pred:
            boxes = convert_boxes(p, self.device)
            box_list.append(boxes)
        result = tuple(box_list)
        return result
    
        #return torch.rand(1, 15, 2, 4) * 10

    def get_binary_road_map(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a cuda tensor with size [batch_size, 800, 800] 
        # from Ben
        cam_samples = samples.transpose(0,1)
        preds = []
        for idx,image_name in enumerate(self.image_names):
            model = self.models[image_name]
            cam_sample = cam_samples[idx]
            preds.append(model(cam_sample) > .5)

        target_road_images = []
        
        for image_idx in range(samples.shape[0]):
            pred_im = [pred[image_idx] for pred in preds]     

            target_road_image = torch.zeros((100,100)).cuda()

            for idx,image in enumerate(self.image_names):
                mask = self.masks[image]
                target_road_image[mask] = target_road_image[mask] + pred_im[idx]
            target_road_image = target_road_image > .5
            resized_trm = resize(target_road_image.cpu(),(800,800))
            target_road_images.append(resized_trm)
        return torch.tensor(target_road_images).cuda()
        
        
       # return torch.rand(1, 800, 800) > 0.5
