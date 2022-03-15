# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 15:14:01 2022

@author: Administrateur
"""

import torch
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import SimpleITK as sitk
import numpy as np
import os
import numpy as np
import skimage
#import skimage.io as io
import skimage.transform as transform
import torch

import numpy as np
import pandas as pd
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import cv2
import matplotlib.pyplot as plt
import skimage 

#import logging
#import traceback
from typing import Optional, Tuple


def kits_normalization(input_image: np.ndarray):
    # first, clip to [-62, 310] (corresponds to 0.5 and 99.5 percentile in the foreground regions)
    # then, subtract 104.9 and divide by 75.3 (corresponds to mean and std in the foreground regions, respectively)
    clip_min = -62
    clip_max = 301
    mean_val = 104.0
    std_val = 75.3
    input_image = np.minimum(np.maximum(input_image, clip_min), clip_max)
    input_image -= mean_val
    input_image /= std_val
    return input_image

def normalize_to_range(input_image: np.ndarray, range: Tuple = (0.0, 1.0)):
    """
    Scales tensor to range
    @param input_image: image of shape (H x W x C)
    @param range:       bounds for normalization
    @return:            normalized image
    """
    max_val = input_image.max()
    min_val = input_image.min()
    if min_val == max_val == 0:
        return input_image
    input_image = input_image - min_val
    input_image = input_image / (max_val - min_val)
    input_image = input_image * (range[1] - range[0])
    input_image = input_image + range[0]
    return input_image

from typing import Iterable

def center_crop(np_image: np.ndarray,
                new_shape: Iterable[int],
                outside_val: float = 0
                ) -> np.ndarray:
    output_image = np.full(new_shape, outside_val, np_image.dtype)

    slices = tuple()
    offsets = tuple()
    for it, sh in enumerate(new_shape):
        size = sh // 2
        if it == 0:
            center = np_image.shape[it] - size
        else:
            center = (np_image.shape[it] // 2)
        start = center - size
        stop = center + size + (sh % 2)

        # computing what area of the original image will be in the cropped output
        slce = slice(max(0, start), min(np_image.shape[it], stop))
        slices += (slce,)

        # computing offset to pad if the crop is partly outside of the scan
        offset = slice(-min(0, start), 2 * size - max(0, (start + 2 * size) - np_image.shape[it]))
        offsets += (offset,)

    output_image[offsets] = np_image[slices]

    return output_image


def pad_image(image: np.ndarray, outer_height: int, outer_width: int, pad_value: Tuple):
    """
    Pastes input image in the middle of a larger one
    @param image:        image of shape (H x W x C)
    @param outer_height: final outer height
    @param outer_width:  final outer width
    @param pad_value:    value for padding around inner image
    @return:             padded image
    """
    inner_height, inner_width = image.shape[0], image.shape[1]
    h_offset = int((outer_height - inner_height) / 2.0)
    w_offset = int((outer_width - inner_width) / 2.0)
    outer_image = np.ones((outer_height, outer_width, 3), dtype=image.dtype) * pad_value
    outer_image[h_offset:h_offset + inner_height, w_offset:w_offset + inner_width, :] = image

    return outer_image


class deeeeset(Dataset):
    def __init__(self,in_out,images,normalized_target_range: Tuple = (0, 1),
                 resize_to: Optional[Tuple] = (256, 256, 110)):
        self.in_out=in_out
        self.images=images
        #self.new_shape=new_shape
        #self.input_data = input_data
        self.normalized_target_range = normalized_target_range
        self.resize_to = resize_to
        
        self.ids=self.in_out['SubjectId']
        self.label=self.in_out['task_1_label']
        self.in_out1=self.in_out.copy()
        
        self.feature=self.in_out1.drop(['SubjectId','task_1_label','task_2_label'],axis=1, inplace=True)

        
        self.path_img=[]
        
        for i in self.ids:
            pth_i=os.path.join(self.images,i)
            pth_ie=os.path.join(pth_i+'/imaging.nii.gz')
            self.path_img.append(pth_ie)
            
        
    def __getitem__(self,idx):
        
        feat=self.in_out1.iloc[idx]
        x_feature=pd.DataFrame(feat).T
        
        x_input=self.path_img[idx]
        read_img=sitk.ReadImage(x_input)
        #get_array_img=sitk.GetArrayFromImage(read_img)
        
        # convert to numpy
        data_npy = sitk.GetArrayFromImage(read_img).astype(np.float32)
            
        # normalize
        #inner_image = normalize_to_range(data_npy, range=self.normalized_target_range)
        inner_image = kits_normalization(data_npy)
        #inner_image=normalize_to_range(inner_image)
        inner_image = normalize_to_range(inner_image, range=self.normalized_target_range)
            
        # resize
        inner_image_height, inner_image_width, inner_image_depth = inner_image.shape[0], inner_image.shape[1], inner_image.shape[2]
            
        # if inner_image_height != 512 or inner_image_width != 512:
        #     # there is one sample in KiTS21 with a 796 number of rows,
        #     # different from all the others. we disregard it for simplicity
        #     return None
        if self.resize_to is not None:
            h_ratio = self.resize_to[0] / inner_image_height
            w_ratio = self.resize_to[1] / inner_image_width
            if h_ratio>=1 and w_ratio>=1:
                resize_ratio_xy = min(h_ratio, w_ratio)
            elif h_ratio<1 and w_ratio<1:
                resize_ratio_xy = max(h_ratio, w_ratio)
            else:
                resize_ratio_xy = 1
            #resize_ratio_z = self.resize_to[2] / inner_image_depth
            if resize_ratio_xy != 1 or inner_image_depth != self.resize_to[2]:
                inner_image = skimage.transform.resize(inner_image,
                                                       output_shape=(int(inner_image_height * resize_ratio_xy),
                                                                     int(inner_image_width * resize_ratio_xy),
                                                                     int(self.resize_to[2])),
                                                       mode='reflect',
                                                       anti_aliasing=True
                                                       )

        image = inner_image
        #input_image = center_crop(input_image, new_shape=new_shape

        # convert image from shape (H x W x D) to shape (D x H x W) 
        image = np.moveaxis(image, -1, 0)

        # add a singleton channel dimension so the image takes the shape (C x D x H x W)
        image = image[np.newaxis, :, :, :]

        # numpy to tensor
        sample = torch.from_numpy(image)
        
        
        # norm_img=get_array_img
        
        y_output=self.label[idx]
        x_f_array=np.array(x_feature)
        x_f_array_t=torch.from_numpy(x_f_array).float()
        x_f_array_t=torch.squeeze(x_f_array_t,axis=0)
        return (sample,x_f_array_t,y_output)
    
    def __len__(self):
        return(len(self.label))
    


#input_image = center_crop(input_image, new_shape=new_shape) 
    
    

filet=pd.read_csv('C:\\Users\\Administrateur\\Desktop\\testfused_model\\trainfold0.csv')
data='C:\\Users\\Administrateur\\Desktop\\data1'

filev=pd.read_csv('C:\\Users\\Administrateur\\Desktop\\testfused_model\\validfold0.csv')

len(filev['task_1_label']==0)
len(filev['task_1_label']==1)

#classes=['NoAT','CanAT']
c1=filet['task_1_label'].value_counts()[0]
#175

#c2=filet['task_1_label'].value_counts()[1]
#65

train_dataset=deeeeset(filet,data)
valid_dataset=deeeeset(filev,data)

print(len(train_dataset))
print(len(valid_dataset))

for i,d in enumerate(train_dataset):
    print(i)
    d,f,l=d
    print(d.shape)
    print(f.shape)
    print(l)