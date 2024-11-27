# leaf disease dataset

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import imp

import os
from io import BytesIO
import json
import logging
import base64
from sys import prefix
import threading
import random
from turtle import left, right
import numpy as np
from typing import Callable, List, Tuple, Union
from PIL import Image,ImageDraw
import torch.utils.data as data
import json
import time
import cv2
import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as T
import copy
import math
from functools import partial
import albumentations as A
import bezier


def bbox_process(bbox):
    x_min = int(bbox[0])
    y_min = int(bbox[1])
    x_max = x_min + int(bbox[2])
    y_max = y_min + int(bbox[3])
    return list(map(int, [x_min, y_min, x_max, y_max]))


def get_tensor(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return torchvision.transforms.Compose(transform_list)

def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)


class LeafDiseasesDataset(data.Dataset):
    def __init__(self,state,dataset_dir,arbitrary_mask_percent=0,**args
        ):
        self.state=state
        self.args=args
        self.dataset_dir=dataset_dir
        self.arbitrary_mask_percent=arbitrary_mask_percent
        self.kernel = np.ones((1, 1), np.uint8)
        self.random_trans=A.Compose([
            A.Resize(height=224,width=224),
            ])
        
        self.images_list=[]
        self.bbox_path_list=[]

        self.imgs_dir = os.path.join(self.dataset_dir, 'images')
        self.bboxs_dir = os.path.join(self.dataset_dir, 'bbox')

        if state == "train":
            plants = os.listdir(os.path.join(self.imgs_dir, 'train'))
            for plant in plants:
                plant_img_dir = os.path.join(self.imgs_dir, 'train', plant)
                plant_bbox_dir = os.path.join(self.bboxs_dir, 'train', plant)
                diseases = os.listdir(plant_img_dir)
                for disease in diseases:
                    self.images_list.append(os.path.join(plant_img_dir, disease))
                    self.bbox_path_list.append(os.path.join(plant_bbox_dir, disease))
        elif state == "validation":
            plants = os.listdir(os.path.join(self.imgs_dir, 'val'))
            for plant in plants:
                plant_img_dir = os.path.join(self.imgs_dir, 'val', plant)
                plant_bbox_dir = os.path.join(self.bboxs_dir, 'val', plant)
                diseases = os.listdir(plant_img_dir)
                for disease in diseases:
                    self.images_list.append(os.path.join(plant_img_dir, disease))
                    self.bbox_path_list.append(os.path.join(plant_bbox_dir, disease))
        else:
            plants = os.listdir(os.path.join(self.imgs_dir, 'test'))
            for plant in plants:
                plant_img_dir = os.path.join(self.imgs_dir, 'test', plant)
                plant_bbox_dir = os.path.join(self.bboxs_dir, 'test', plant)
                diseases = os.listdir(plant_img_dir)
                for disease in diseases:
                    self.images_list.append(os.path.join(plant_img_dir, disease))
                    self.bbox_path_list.append(os.path.join(plant_bbox_dir, disease))

        self.length = len(self.images_list)
 
    def __getitem__(self, index):
        bbox_path = self.bbox_path_list[index]
        img_path = self.images_list[index]

        bbox_disease=[]
        with open(bbox_path) as f:
            # read first line that is disease region
            line=f.readline()
            line_split=line.strip('\n').split(" ")
            for i in range(4):
                bbox_disease.append(int(float(line_split[i])))

        bbox=bbox_disease
        img_p = Image.open(img_path).convert("RGB")

        ### Get reference image
        bbox_pad=copy.copy(bbox)
        bbox_pad[0]=bbox[0]-min(10,bbox[0]-0)
        bbox_pad[1]=bbox[1]-min(10,bbox[1]-0)
        bbox_pad[2]=bbox[2]+min(10,img_p.size[0]-bbox[2])
        bbox_pad[3]=bbox[3]+min(10,img_p.size[1]-bbox[3])
        img_p_np=cv2.imread(img_path)
        img_p_np = cv2.cvtColor(img_p_np, cv2.COLOR_BGR2RGB)
        
        ref_image_tensor=img_p_np[bbox_pad[1]:bbox_pad[3],bbox_pad[0]:bbox_pad[2],:]
        ref_image_tensor=self.random_trans(image=ref_image_tensor)
        ref_image_tensor=Image.fromarray(ref_image_tensor["image"])
        ref_image_tensor=get_tensor_clip()(ref_image_tensor)

        ### Generate mask
        ### TODO: use maskrcnn mask
        image_tensor = get_tensor()(img_p)
        W,H = img_p.size

        extended_bbox=copy.copy(bbox)
        left_freespace=bbox[0]-0
        right_freespace=W-bbox[2]
        up_freespace=bbox[1]-0
        down_freespace=H-bbox[3]
        extended_bbox[0]=bbox[0]-random.randint(0,int(0.4*left_freespace))
        extended_bbox[1]=bbox[1]-random.randint(0,int(0.4*up_freespace))
        extended_bbox[2]=bbox[2]+random.randint(0,int(0.4*right_freespace))
        extended_bbox[3]=bbox[3]+random.randint(0,int(0.4*down_freespace))

        mask_img=np.zeros((H,W))
        mask_img[extended_bbox[1]:extended_bbox[3],extended_bbox[0]:extended_bbox[2]]=1
        mask_img=Image.fromarray(mask_img)
        mask_tensor=1-get_tensor(normalize=False, toTensor=True)(mask_img)

        ### Crop square image
        if W > H:
            left_most=extended_bbox[2]-H
            if left_most <0:
                left_most=0
            right_most=extended_bbox[0]+H
            if right_most > W:
                right_most=W
            right_most=right_most-H
            if right_most<= left_most:
                image_tensor_cropped=image_tensor
                mask_tensor_cropped=mask_tensor
            else:
                left_pos=random.randint(left_most,right_most) 
                free_space=min(extended_bbox[1]-0,extended_bbox[0]-left_pos,left_pos+H-extended_bbox[2],H-extended_bbox[3])
                random_free_space=random.randint(0,int(0.6*free_space))
                image_tensor_cropped=image_tensor[:,0+random_free_space:H-random_free_space,left_pos+random_free_space:left_pos+H-random_free_space]
                mask_tensor_cropped=mask_tensor[:,0+random_free_space:H-random_free_space,left_pos+random_free_space:left_pos+H-random_free_space]
        
        elif  W < H:
            upper_most=extended_bbox[3]-W
            if upper_most <0:
                upper_most=0
            lower_most=extended_bbox[1]+W
            if lower_most > H:
                lower_most=H
            lower_most=lower_most-W
            if lower_most<=upper_most:
                image_tensor_cropped=image_tensor
                mask_tensor_cropped=mask_tensor
            else:
                upper_pos=random.randint(upper_most,lower_most) 
                free_space=min(extended_bbox[1]-upper_pos,extended_bbox[0]-0,W-extended_bbox[2],upper_pos+W-extended_bbox[3])
                random_free_space=random.randint(0,int(0.6*free_space))
                image_tensor_cropped=image_tensor[:,upper_pos+random_free_space:upper_pos+W-random_free_space,random_free_space:W-random_free_space]
                mask_tensor_cropped=mask_tensor[:,upper_pos+random_free_space:upper_pos+W-random_free_space,random_free_space:W-random_free_space]
        else:
            image_tensor_cropped=image_tensor
            mask_tensor_cropped=mask_tensor

        image_tensor_resize=T.Resize([self.args['image_size'],self.args['image_size']])(image_tensor_cropped)
        mask_tensor_resize=T.Resize([self.args['image_size'],self.args['image_size']])(mask_tensor_cropped)
        inpaint_tensor_resize=image_tensor_resize*mask_tensor_resize

        return {"GT":image_tensor_resize,"inpaint_image":inpaint_tensor_resize,"inpaint_mask":mask_tensor_resize,"ref_imgs":ref_image_tensor}

    def __len__(self):
        return self.length



