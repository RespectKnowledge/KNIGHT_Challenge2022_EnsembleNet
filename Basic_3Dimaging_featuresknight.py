# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 21:03:37 2022

@author: Abdul Qayyum
"""
## basic image processing features for knight challenege
import os
import nibabel as nib
import pyfeats
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pyfeats import *

#img=nib.load(pathf)
#img2=img.get_fdata()


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
import skimage
import SimpleITK as sitk

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


# read_img=sitk.ReadImage(pathf)
# #get_array_img=sitk.GetArrayFromImage(read_img)

# # convert to numpy
# data_npy = sitk.GetArrayFromImage(read_img).astype(np.float32)
# inner_image = kits_normalization(data_npy)
# out_im=preprocessing(inner_image)

def preprocessing(img,resize_to=(256,256,110)):
    inner_image = kits_normalization(data_npy)
    #inner_image=normalize_to_range(inner_image)
    #inner_image = normalize_to_range(inner_image, range=normalized_target_range)
        
    # resize
    inner_image_height, inner_image_width, inner_image_depth = inner_image.shape[0], inner_image.shape[1], inner_image.shape[2]
        
    if resize_to is not None:
        h_ratio = resize_to[0] / inner_image_height
        w_ratio = resize_to[1] / inner_image_width
        if h_ratio>=1 and w_ratio>=1:
            resize_ratio_xy = min(h_ratio, w_ratio)
        elif h_ratio<1 and w_ratio<1:
            resize_ratio_xy = max(h_ratio, w_ratio)
        else:
            resize_ratio_xy = 1
        #resize_ratio_z = self.resize_to[2] / inner_image_depth
        if resize_ratio_xy != 1 or inner_image_depth != resize_to[2]:
            inner_image = skimage.transform.resize(inner_image,
                                                   output_shape=(int(inner_image_height * resize_ratio_xy),
                                                                 int(inner_image_width * resize_ratio_xy),
                                                                 int(resize_to[2])),
                                                   mode='reflect',
                                                   anti_aliasing=True
                                                   )
    
    
    return inner_image

from scipy import stats
path_new='C:\\Users\\data'
lsitd=os.listdir(path_new)
features={}
for i in lsitd:
    print(i)
    #pathm=os.path.join(path_new,i)
    pathm=os.path.join(path_new,i+'\imaging.nii.gz')
    print(pathm)
    read_img=sitk.ReadImage(pathm)
    #get_array_img=sitk.GetArrayFromImage(read_img)

    # convert to numpy
    data_npy = sitk.GetArrayFromImage(read_img).astype(np.float32)
    inner_image = kits_normalization(data_npy)
    out_im=preprocessing(inner_image)
    # flatten 3d array
    #img_data = self._read_file(file_path)
    data = out_im.reshape(-1)
    # create features
    data_mean = data.mean()
    data_std = data.std()
    intensive_data = data[data > data_mean]
    more_intensive_data = data[data > data_mean + data_std]
    non_intensive_data = data[data < data_mean]
                
    data_skew = stats.skew(data)
    data_kurtosis = stats.kurtosis(data)
    intensive_skew = stats.skew(intensive_data)
    non_intensive_skew = stats.skew(non_intensive_data)
                
    data_diff = np.diff(data)
    data_type=str(i)
    features={}          
    # write new features in df
    features['SubjectId'] = i
    features[f'{data_type}_skew'] = data_skew,
    features[f'{data_type}_kurtosis'] = data_kurtosis,
    features[f'{data_type}_diff_skew'] = stats.skew(data_diff),
    features[f'{data_type}_intensive_dist'] = intensive_data.shape[0],
    features[f'{data_type}_intensive_skew'] = intensive_skew,
    features[f'{data_type}_non_intensive_dist'] = non_intensive_data.shape[0],
    features[f'{data_type}_non_intensive_skew'] = non_intensive_skew,
    #features[f'{data_type}_intensive_non_intensive_mean_ratio'] = intensive_data.mean() / non_intensive_data.mean(),
    #features[f'{data_type}_intensive_non_intensive_std_ratio'] = intensive_data.std() / non_intensive_data.std(),
    features[f'{data_type}_data_intensive_skew_difference'] = data_skew - intensive_skew,
    features[f'{data_type}_data_non_intensive_skew_difference'] = data_skew - non_intensive_skew,
    features[f'{data_type}_more_intensive_dist'] = more_intensive_data.shape[0],
    #break
#     #df8.loc[i,'subject_id']=i
#     features1={}
#     features2={}
#     features3={}
#     features1['A_FOS'] = fos(out_im,None)
#     df8.loc[i,:]=features1['A_FOS'][0]
#     #featuresf={}
#     features2['A_GLCM'] = glcm_features(out_im, ignore_zeros=True)
    
#     features3['C_Histogram'] = histogram(out_im, None, bins=32) # can be used for 3D
#     dfc.loc[i,:]=features2['A_GLCM'][0]
#     dfd.loc[i,:]=features2['A_GLCM'][1]
#     dfbins.loc[i,:]=features3['C_Histogram'][0]
#     #break
# #f1,f2=featuresf['C_Histogram'] 
import pandas as pd
ff=pd.DataFrame(features)
k=features.items()
dff=['SubjectId','skew']