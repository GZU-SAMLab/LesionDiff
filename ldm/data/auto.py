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

from groundingdino.util.inference import load_model, load_image, predict, annotate, Model
from torchvision.ops import box_convert

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


def change_unqualified_bbox(left, top, right, bottom, standard, size):
    if right - left < standard:
        if left - 50 > 0:
            left -= 50
        else:
            left = 0
        if right + 50 < size:
            right += 50
        else:
            right = size
    if bottom - top < standard:
        if top - 50 > 0:
            top -= 50
        else:
            top = 0
        if bottom + 50 < size:
            bottom += 50
        else:
            bottom = size
    return left, top, right, bottom

class LeafDiseasesDataset(data.Dataset):
    def __init__(self,
            state,
            dataset_dir,
            arbitrary_mask_percent=0,
            type = "whole",
            CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            CHECKPOINT_PATH = "GroundingDINO/models/groundingdino_swint_ogc.pth",
            DEVICE = "cpu",
            # TEXT_PROMPT = "spot",
            TEXT_PROMPT = "leaf",
            BOX_TRESHOLD = 0.35,
            TEXT_TRESHOLD = 0.25,
            FP16_INFERENCE = True,
            **args
        ):
        self.state=state
        self.args=args
        self.dataset_dir=dataset_dir
        self.arbitrary_mask_percent=arbitrary_mask_percent
        self.type=type
        self.kernel = np.ones((1, 1), np.uint8)
        
        self.random_trans=A.Compose([
            A.Resize(height=224,width=224),
        ])
        
        self.images_list=[]
       
        if state == "train":
            plants = os.listdir(os.path.join(self.dataset_dir, 'train'))
            for plant in plants:
                plant_dir = os.path.join(self.dataset_dir, 'train', plant)
                diseases = os.listdir(plant_dir)
                for disease in diseases:
                    self.images_list.append(os.path.join(plant_dir, disease))
        elif state == "validation":
            plants = os.listdir(os.path.join(self.dataset_dir, 'val'))
            for plant in plants:
                plant_dir = os.path.join(self.dataset_dir, 'val', plant)
                diseases = os.listdir(plant_dir)
                for disease in diseases:
                    self.images_list.append(os.path.join(plant_dir, disease))
        else:
            plants = os.listdir(os.path.join(self.dataset_dir, 'test'))
            for plant in plants:
                plant_dir = os.path.join(self.dataset_dir, 'test', plant)
                diseases = os.listdir(plant_dir)
                for disease in diseases:
                    self.images_list.append(os.path.join(plant_dir, disease))
        
        self.length=len(self.images_list)
 
        # load groundingding model
        self.gd_config_path = CONFIG_PATH
        self.gd_checkpoint_path = CHECKPOINT_PATH
        self.gd_device = DEVICE
        self.gd_text_prompt = TEXT_PROMPT
        self.gd_box_threshold = BOX_TRESHOLD
        self.gd_text_threshold = TEXT_TRESHOLD
        self.gd_fp16_inference = FP16_INFERENCE

        self.gd_model = load_model(CONFIG_PATH, CHECKPOINT_PATH)
        if self.gd_fp16_inference:
            self.gd_model = self.gd_model.float()

    def __getitem__(self, index):
        image_path = self.images_list[index]

        ### Get bbox
        image_source, image = load_image(image_path)
        if self.gd_fp16_inference:
            image = image.float()
        boxes, logits, phrases = predict(
            model=self.gd_model,
            image=image,
            caption=self.gd_text_prompt,
            box_threshold=self.gd_box_threshold,
            text_threshold=self.gd_text_threshold,
            device=self.gd_device,
        )
        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        bbox_disease = np.array([156, 156, 356, 356])
        if xyxy.shape[0] == 0:
            bbox_disease = np.array([156, 156, 356, 356])
        else:
            if self.type == "one":
                bbox_disease = xyxy[0].astype(int)
                bbox_disease = [*change_unqualified_bbox(bbox_disease[0], bbox_disease[1], bbox_disease[2], bbox_disease[3], 224, 512)]
            else:
                # find the largest bbox
                left, top, right, bottom = xyxy[0].astype(int)
                for box in xyxy:
                    box = box.astype(int)
                    if box[0] < left:
                        left = box[0]
                    if box[1] < top:
                        top = box[1]
                    if box[2] > right:
                        right = box[2]
                    if box[3] > bottom:
                        bottom = box[3]
                bbox_disease = [*change_unqualified_bbox(left=left, top=top, right=right, bottom=bottom, standard=224, size=512)]
        bbox = bbox_disease
        ref_bbox = bbox_disease
        img_p = Image.open(image_path).convert("RGB")
        
        ### Get reference image
        bbox_pad=copy.copy(ref_bbox)
        bbox_pad[0]=ref_bbox[0]-min(10,ref_bbox[0]-0)
        bbox_pad[1]=ref_bbox[1]-min(10,ref_bbox[1]-0)
        bbox_pad[2]=ref_bbox[2]+min(10,img_p.size[0]-ref_bbox[2])
        bbox_pad[3]=ref_bbox[3]+min(10,img_p.size[1]-ref_bbox[3])
        img_p_np=cv2.imread(image_path)
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



