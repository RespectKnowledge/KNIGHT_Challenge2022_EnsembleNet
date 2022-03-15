# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 14:35:26 2021

@author: Abdul Qayyum
"""

#%% medical imageing features for radiomics

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




# read_img=sitk.ReadImage(pathf)
# #get_array_img=sitk.GetArrayFromImage(read_img)

# # convert to numpy
# data_npy = sitk.GetArrayFromImage(read_img).astype(np.float32)
# inner_image = kits_normalization(data_npy)
# out_im=preprocessing(inner_image)
# out_im1=np.swapaxes(out_im,2,0)

# def maxprojection(img):
#     ind=img.argmax(axis=0)
#     a1,a2=np.indices(ind.shape)
#     maxp=img[ind,a1,a2]
#     return maxp

# plt.imshow(out_im1[3,:,:])
# maximg=maxprojection(out_im1)
# import matplotlib.pyplot as plt
# plt.imshow(maximg)

# image=out_im
# image=maximg

# features = {}
# features['A_FOS'] = fos(image,None)

import pandas as pd
#data1=pd.DataFrame(y)
#path_new='C:\\Users\\mirazzak\\knight_test_data_corrected\\images'
path_new='C:\\Users\\mirazzak\\data'

lsitd=os.listdir(path_new)
Features_name=['FOS_Mean',
 'FOS_Variance',
 'FOS_Median',
 'FOS_Mode',
 'FOS_Skewness',
 'FOS_Kurtosis',
 'FOS_Energy',
 'FOS_Entropy',
 'FOS_MinimalGrayLevel',
 'FOS_MaximalGrayLevel',
 'FOS_CoefficientOfVariation',
 'FOS_10Percentile',
 'FOS_25Percentile',
 'FOS_75Percentile',
 'FOS_90Percentile',
 'FOS_HistogramWidth']
df8=pd.DataFrame(Features_name).T

#a,b,c,d=features['A_GLCM']
featurec=['GLCM_ASM_Mean',
 'GLCM_Contrast_Mean',
 'GLCM_Correlation_Mean',
 'GLCM_SumOfSquaresVariance_Mean',
 'GLCM_InverseDifferenceMoment_Mean',
 'GLCM_SumAverage_Mean',
 'GLCM_SumVariance_Mean',
 'GLCM_SumEntropy_Mean',
 'GLCM_Entropy_Mean',
 'GLCM_DifferenceVariance_Mean',
 'GLCM_DifferenceEntropy_Mean',
 'GLCM_Information1_Mean',
 'GLCM_Information2_Mean',
 'GLCM_MaximalCorrelationCoefficient_Mean']
featuresd=['GLCM_ASM_Range',
 'GLCM_Contrast_Range',
 'GLCM_Correlation_Range',
 'GLCM_SumOfSquaresVariance_Range',
 'GLCM_InverseDifferenceMoment_Range',
 'GLCM_SumAverage_Range',
 'GLCM_SumVariance_Range',
 'GLCM_SumEntropy_Range',
 'GLCM_Entropy_Range',
 'GLCM_DifferenceVariance_Range',
 'GLCM_DifferenceEntropy_Range',
 'GLCM_Information1_Range',
 'GLCM_Information2_Range',
 'GLCM_MaximalCorrelationCoefficient_Range']
ffreq=['Histogram_bin_0',
 'Histogram_bin_1',
 'Histogram_bin_2',
 'Histogram_bin_3',
 'Histogram_bin_4',
 'Histogram_bin_5',
 'Histogram_bin_6',
 'Histogram_bin_7',
 'Histogram_bin_8',
 'Histogram_bin_9',
 'Histogram_bin_10',
 'Histogram_bin_11',
 'Histogram_bin_12',
 'Histogram_bin_13',
 'Histogram_bin_14',
 'Histogram_bin_15',
 'Histogram_bin_16',
 'Histogram_bin_17',
 'Histogram_bin_18',
 'Histogram_bin_19',
 'Histogram_bin_20',
 'Histogram_bin_21',
 'Histogram_bin_22',
 'Histogram_bin_23',
 'Histogram_bin_24',
 'Histogram_bin_25',
 'Histogram_bin_26',
 'Histogram_bin_27',
 'Histogram_bin_28',
 'Histogram_bin_29',
 'Histogram_bin_30',
 'Histogram_bin_31']

dfc=pd.DataFrame(featurec).T
dfd=pd.DataFrame(featuresd).T
#ffreq
dfbins=pd.DataFrame(ffreq).T

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
    #df8.loc[i,'subject_id']=i
    features1={}
    features2={}
    features3={}
    features1['A_FOS'] = fos(out_im,None)
    df8.loc[i,:]=features1['A_FOS'][0]
    #featuresf={}
    features2['A_GLCM'] = glcm_features(out_im, ignore_zeros=True)
    
    features3['C_Histogram'] = histogram(out_im, None, bins=32) # can be used for 3D
    dfc.loc[i,:]=features2['A_GLCM'][0]
    dfd.loc[i,:]=features2['A_GLCM'][1]
    dfbins.loc[i,:]=features3['C_Histogram'][0]
    #break
#f1,f2=featuresf['C_Histogram'] 
#%%
testdf=df8.copy()

firstdf=testdf.reset_index()
firstdf=firstdf.drop(0)
old_columns=(firstdf.columns)
#new_list=d2f
Features_name=['SubjectId','FOS_Mean',
 'FOS_Variance',
 'FOS_Median',
 'FOS_Mode',
 'FOS_Skewness',
 'FOS_Kurtosis',
 'FOS_Energy',
 'FOS_Entropy',
 'FOS_MinimalGrayLevel',
 'FOS_MaximalGrayLevel',
 'FOS_CoefficientOfVariation',
 'FOS_10Percentile',
 'FOS_25Percentile',
 'FOS_75Percentile',
 'FOS_90Percentile',
 'FOS_HistogramWidth']
new_list=Features_name

firstdf.rename(columns={old_columns[idx]: name for (idx,name) in enumerate(new_list)}, inplace=True)
firstdf.to_csv('A_FOS_features_train.csv',index=False)
#%%dfcc
testdf=dfc.copy()

firstdf=testdf.reset_index()
firstdf=firstdf.drop(0)
old_columns=(firstdf.columns)
#new_list=d2f
featurec=['SubjectId','GLCM_ASM_Mean',
 'GLCM_Contrast_Mean',
 'GLCM_Correlation_Mean',
 'GLCM_SumOfSquaresVariance_Mean',
 'GLCM_InverseDifferenceMoment_Mean',
 'GLCM_SumAverage_Mean',
 'GLCM_SumVariance_Mean',
 'GLCM_SumEntropy_Mean',
 'GLCM_Entropy_Mean',
 'GLCM_DifferenceVariance_Mean',
 'GLCM_DifferenceEntropy_Mean',
 'GLCM_Information1_Mean',
 'GLCM_Information2_Mean',
 'GLCM_MaximalCorrelationCoefficient_Mean']
new_list=featurec

firstdf.rename(columns={old_columns[idx]: name for (idx,name) in enumerate(new_list)}, inplace=True)
firstdf.to_csv('A_GLCM1_train.csv',index=False)

#%%
testdf=dfd.copy()

firstdf=testdf.reset_index()
firstdf=firstdf.drop(0)
old_columns=(firstdf.columns)
#new_list=d2f
featuresd=['SubjectId','GLCM_ASM_Range',
 'GLCM_Contrast_Range',
 'GLCM_Correlation_Range',
 'GLCM_SumOfSquaresVariance_Range',
 'GLCM_InverseDifferenceMoment_Range',
 'GLCM_SumAverage_Range',
 'GLCM_SumVariance_Range',
 'GLCM_SumEntropy_Range',
 'GLCM_Entropy_Range',
 'GLCM_DifferenceVariance_Range',
 'GLCM_DifferenceEntropy_Range',
 'GLCM_Information1_Range',
 'GLCM_Information2_Range',
 'GLCM_MaximalCorrelationCoefficient_Range']
new_list=featuresd

firstdf.rename(columns={old_columns[idx]: name for (idx,name) in enumerate(new_list)}, inplace=True)
firstdf.to_csv('A_GLCM2_train.csv',index=False)
#%%
ffreq=['SubjectId','Histogram_bin_0',
 'Histogram_bin_1',
 'Histogram_bin_2',
 'Histogram_bin_3',
 'Histogram_bin_4',
 'Histogram_bin_5',
 'Histogram_bin_6',
 'Histogram_bin_7',
 'Histogram_bin_8',
 'Histogram_bin_9',
 'Histogram_bin_10',
 'Histogram_bin_11',
 'Histogram_bin_12',
 'Histogram_bin_13',
 'Histogram_bin_14',
 'Histogram_bin_15',
 'Histogram_bin_16',
 'Histogram_bin_17',
 'Histogram_bin_18',
 'Histogram_bin_19',
 'Histogram_bin_20',
 'Histogram_bin_21',
 'Histogram_bin_22',
 'Histogram_bin_23',
 'Histogram_bin_24',
 'Histogram_bin_25',
 'Histogram_bin_26',
 'Histogram_bin_27',
 'Histogram_bin_28',
 'Histogram_bin_29',
 'Histogram_bin_30',
 'Histogram_bin_31']
testdf=dfbins.copy()

firstdf=testdf.reset_index()
firstdf=firstdf.drop(0)
old_columns=(firstdf.columns)
#new_list=d2f
new_list=ffreq

firstdf.rename(columns={old_columns[idx]: name for (idx,name) in enumerate(new_list)}, inplace=True)
firstdf.to_csv('A_Histogram_train.csv',index=False)
#%%

#%%
df2=pd.DataFrame(lsitd,columns=['SubjectId'])
dff=df8

#%%
df2=pd.DataFrame(lsitd,columns=['SubjectId'])
dff=df8
results1=pd.concat([df8,df2],axis=0,ignore_index=True)
df=df8.merge(df2,how='left',left_on=0,right_on=0)
df3=df8.rename()
df8.index
df8.rename(index={'subject':'0'},inplace=True)
df8.index=Features_name=['Sub','FOS_Mean',
 'FOS_Variance',
 'FOS_Median',
 'FOS_Mode',
 'FOS_Skewness',
 'FOS_Kurtosis',
 'FOS_Energy',
 'FOS_Entropy',
 'FOS_MinimalGrayLevel',
 'FOS_MaximalGrayLevel',
 'FOS_CoefficientOfVariation',
 'FOS_10Percentile',
 'FOS_25Percentile',
 'FOS_75Percentile',
 'FOS_90Percentile',
 'FOS_HistogramWidth'].T
df8.columns
dd=df8.reset_index()
dd1=dd.drop(0)
dd2=dd1.reset_index()
dd3=dd1.rename(columns={'index':'SubjectId'})
dd3.to_csv('fos_features1.csv',index=False)
#%%

#%%


features={}
features['A_GLCM'] = glcm_features(image, ignore_zeros=True)


# features['A_GLCM'] = glcm_features(image, ignore_zeros=True)
# features['C_Histogram'] = histogram(image, None, bins=32) # can be used for 3D
# #features['A_GLDS'] = glds_features(image,None)
#features['A_NGTDM'] = ngtdm_features(image,None, d=1)
#features['A_SFM'] = sfm_features(image,None,Lr=4, Lc=4)
#features['A_LTE'] = lte_measures(image,None,l=7)
#features['A_FDTA'] = fdta(image,None)
#features['A_GLRLM'] = glrlm_features(image, None,Ng=256)
#features['A_FPS'] = fps(image, None)
#features['A_Shape_Parameters'] = shape_parameters(image,None, perimeter=1, pixels_per_mm2=1)
#features['A_HOS'] = hos_features(image, th=[135,140])
#features['A_LBP'] = lbp_features(image, image, P=[8,16,24], R=[1,2,3])
#features['A_GLSZM'] = glszm_features(image,None)
for x,y in features.items():
    print(x)
    print(y)
a,c=y
d,e=features['A_FOS']
df=pd.DataFrame(e)
for i in range(0,len(e)):
    df.loc[:,i]=d
df.loc[:,]=d

Features_name=['FOS_Mean',
 'FOS_Variance',
 'FOS_Median',
 'FOS_Mode',
 'FOS_Skewness',
 'FOS_Kurtosis',
 'FOS_Energy',
 'FOS_Entropy',
 'FOS_MinimalGrayLevel',
 'FOS_MaximalGrayLevel',
 'FOS_CoefficientOfVariation',
 'FOS_10Percentile',
 'FOS_25Percentile',
 'FOS_75Percentile',
 'FOS_90Percentile',
 'FOS_HistogramWidth']
df7=pd.DataFrame(Features_name).T
for d,i in enumerate(features['A_FOS']):
    print(d)
    print(i)
    df7.loc[d,:]=features['A_FOS'][0]
features['A_FOS'][0]


# features1 = {}
# #features['A_FOS'] = fos(image,)
# features1['A_GLCM'] = glcm_features(img2, ignore_zeros=True)
# features = {}
df6.loc[1,:]=features['A_FOS'][0]
# features['A_FOS'] = fos(img2, None)
import pandas as pd
data1=pd.DataFrame(y)
path_new='C:\\Users\\mirazzak\\testingdataset'

lsitd=os.listdir(path_new)
Features_name=['FOS_Mean',
 'FOS_Variance',
 'FOS_Median',
 'FOS_Mode',
 'FOS_Skewness',
 'FOS_Kurtosis',
 'FOS_Energy',
 'FOS_Entropy',
 'FOS_MinimalGrayLevel',
 'FOS_MaximalGrayLevel',
 'FOS_CoefficientOfVariation',
 'FOS_10Percentile',
 'FOS_25Percentile',
 'FOS_75Percentile',
 'FOS_90Percentile',
 'FOS_HistogramWidth']
df8=pd.DataFrame(Features_name).T
for i in lsitd:
    print(i)
    pathm=os.path.join(path_new,i+'\imaging.nii.gz')
    print(pathm)
    read_img=sitk.ReadImage(pathf)
    #get_array_img=sitk.GetArrayFromImage(read_img)

    # convert to numpy
    data_npy = sitk.GetArrayFromImage(read_img).astype(np.float32)
    inner_image = kits_normalization(data_npy)
    out_im=preprocessing(inner_image)
    features={}
    features['A_FOS'] = fos(image,None)
    df8.loc[i,:]=features['A_FOS'][0]


#%%



import pyfeats
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pyfeats import *

path='C:\\Users\\Administrateur\\Downloads\\pyfeats-main\\pyfeats-main\\demo\\data\\f.png'
image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
path1='C:\\Users\\Administrateur\\Downloads\\pyfeats-main\\pyfeats-main\\demo\\data\\mask.png'
mask = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)

plt.imshow(image, cmap='gray')
plt.title('Initial Image')
plt.show()
features, labels = fos(image,None)
#%% A1. Texture features
features = {}
features['A_FOS'] = fos(image, None)
features['A_GLCM'] = glcm_features(image, ignore_zeros=True)
features['A_GLDS'] = glds_features(image, None, Dx=[0,1,1,1], Dy=[1,1,0,-1])
features['A_NGTDM'] = ngtdm_features(image, None, d=1)
features['A_SFM'] = sfm_features(image, None, Lr=4, Lc=4)
features['A_LTE'] = lte_measures(image, None, l=7)
features['A_FDTA'] = fdta(image, None, s=3)
features['A_GLRLM'] = glrlm_features(image, None, Ng=256)
features['A_FPS'] = fps(image, None)
features['A_Shape_Parameters'] = shape_parameters(image, None, perimeter=1, pixels_per_mm2=1)
features['A_HOS'] = hos_features(image, th=[135,140])
features['A_LBP'] = lbp_features(image, image, P=[8,16,24], R=[1,2,3])
features['A_GLSZM'] = glszm_features(image, None)
image_name ='ultrasound.bmp'
plot_sinogram(image, image_name)

#%% B. Morphological features
#features = {}
#features['B_Morphological_Grayscale_pdf'], features['B_Morphological_Grayscale_cdf'] = grayscale_morphology_features(image, N=30)
#features['B_Morphological_Binary_L_pdf'], features['B_Morphological_Binary_M_pdf'], features['B_Morphological_Binary_H_pdf'], features['B_Morphological_Binary_L_cdf'], \
#features['B_Morphological_Binary_M_cdf'], features['B_Morphological_Binary_H_cdf'] = multilevel_binary_morphology_features(image, None, N=30, thresholds=[25,50])

#plot_pdf_cdf(features['B_Morphological_Grayscale_pdf'], features['B_Morphological_Grayscale_cdf'], image_name)
#plot_pdfs_cdfs(features['B_Morphological_Binary_L_pdf'], features['B_Morphological_Binary_M_pdf'], features['B_Morphological_Binary_H_pdf'], features['B_Morphological_Binary_L_cdf'], features['B_Morphological_Binary_M_cdf'], features['B_Morphological_Binary_H_cdf'])

#%% C. Histogram Based features
features['C_Histogram'] = histogram(image, None, bins=32) # can be used for 3D
features['C_MultiregionHistogram'] = multiregion_histogram(image, None, bins=32, num_eros=3, square_size=3)
features['C_Correlogram'] = correlogram(image,None, bins_digitize=32, bins_hist=32, flatten=True)

#plot_histogram(image,None, bins=32, name=image_name)
#plot_correlogram(image, None, bins_digitize=32, bins_hist=32, name=image_name)

#%% D. Multi-Scale features

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pyfeats import *
path='C:\\Users\\mirazzak\\data\\case_00000'

pathf=os.path.join(path,'imaging.nii.gz')

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




# read_img=sitk.ReadImage(pathf)
# #get_array_img=sitk.GetArrayFromImage(read_img)

# # convert to numpy
# data_npy = sitk.GetArrayFromImage(read_img).astype(np.float32)
# inner_image = kits_normalization(data_npy)
# out_im=preprocessing(inner_image)
# out_im1=np.swapaxes(out_im,2,0)

def maxprojection(img):
    ind=img.argmax(axis=0)
    a1,a2=np.indices(ind.shape)
    maxp=img[ind,a1,a2]
    return maxp

#plt.imshow(out_im1[3,:,:])
#maximg=maxprojection(out_im1)
# import matplotlib.pyplot as plt
# plt.imshow(maximg)

# image=out_im
# image=maximg

# features = {}
# features['A_FOS'] = fos(image,None)

import pandas as pd
#data1=pd.DataFrame(y)
#path_new='C:\\Users\\mirazzak\\data2\\data'

#lsitd=os.listdir(path_new)

#d1,d2=features['D_DWT']
d2f=['DWT_bior3.3_level_1_da_mean',
 'DWT_bior3.3_level_1_da_std',
 'DWT_bior3.3_level_1_dd_mean',
 'DWT_bior3.3_level_1_dd_std',
 'DWT_bior3.3_level_1_ad_mean',
 'DWT_bior3.3_level_1_ad_std',
 'DWT_bior3.3_level_2_da_mean',
 'DWT_bior3.3_level_2_da_std',
 'DWT_bior3.3_level_2_dd_mean',
 'DWT_bior3.3_level_2_dd_std',
 'DWT_bior3.3_level_2_ad_mean',
 'DWT_bior3.3_level_2_ad_std',
 'DWT_bior3.3_level_3_da_mean',
 'DWT_bior3.3_level_3_da_std',
 'DWT_bior3.3_level_3_dd_mean',
 'DWT_bior3.3_level_3_dd_std',
 'DWT_bior3.3_level_3_ad_mean',
 'DWT_bior3.3_level_3_ad_std']

d2ff=pd.DataFrame(d2f).T
#ds1,ds2=features['D_SWT']
ds2f=['SWT_bior3.3_level_1_h_mean',
 'SWT_bior3.3_level_1_h_std',
 'SWT_bior3.3_level_1_v_mean',
 'SWT_bior3.3_level_1_v_std',
 'SWT_bior3.3_level_1_d_mean',
 'SWT_bior3.3_level_1_d_std',
 'SWT_bior3.3_level_2_h_mean',
 'SWT_bior3.3_level_2_h_std',
 'SWT_bior3.3_level_2_v_mean',
 'SWT_bior3.3_level_2_v_std',
 'SWT_bior3.3_level_2_d_mean',
 'SWT_bior3.3_level_2_d_std',
 'SWT_bior3.3_level_3_h_mean',
 'SWT_bior3.3_level_3_h_std',
 'SWT_bior3.3_level_3_v_mean',
 'SWT_bior3.3_level_3_v_std',
 'SWT_bior3.3_level_3_d_mean',
 'SWT_bior3.3_level_3_d_std']
ds2ff=pd.DataFrame(ds2f).T

#dwp1,dwp2=features['D_WP']
dwpf=['WP_coif1_aah_mean',
 'WP_coif1_aah_std',
 'WP_coif1_aav_mean',
 'WP_coif1_aav_std',
 'WP_coif1_aad_mean',
 'WP_coif1_aad_std',
 'WP_coif1_aha_mean',
 'WP_coif1_aha_std',
 'WP_coif1_ahh_mean',
 'WP_coif1_ahh_std',
 'WP_coif1_ahv_mean',
 'WP_coif1_ahv_std',
 'WP_coif1_ahd_mean',
 'WP_coif1_ahd_std',
 'WP_coif1_ava_mean',
 'WP_coif1_ava_std',
 'WP_coif1_avh_mean',
 'WP_coif1_avh_std',
 'WP_coif1_avv_mean',
 'WP_coif1_avv_std',
 'WP_coif1_avd_mean',
 'WP_coif1_avd_std',
 'WP_coif1_ada_mean',
 'WP_coif1_ada_std',
 'WP_coif1_adh_mean',
 'WP_coif1_adh_std',
 'WP_coif1_adv_mean',
 'WP_coif1_adv_std',
 'WP_coif1_add_mean',
 'WP_coif1_add_std',
 'WP_coif1_haa_mean',
 'WP_coif1_haa_std',
 'WP_coif1_hah_mean',
 'WP_coif1_hah_std',
 'WP_coif1_hav_mean',
 'WP_coif1_hav_std',
 'WP_coif1_had_mean',
 'WP_coif1_had_std',
 'WP_coif1_hha_mean',
 'WP_coif1_hha_std',
 'WP_coif1_hhh_mean',
 'WP_coif1_hhh_std',
 'WP_coif1_hhv_mean',
 'WP_coif1_hhv_std',
 'WP_coif1_hhd_mean',
 'WP_coif1_hhd_std',
 'WP_coif1_hva_mean',
 'WP_coif1_hva_std',
 'WP_coif1_hvh_mean',
 'WP_coif1_hvh_std',
 'WP_coif1_hvv_mean',
 'WP_coif1_hvv_std',
 'WP_coif1_hvd_mean',
 'WP_coif1_hvd_std',
 'WP_coif1_hda_mean',
 'WP_coif1_hda_std',
 'WP_coif1_hdh_mean',
 'WP_coif1_hdh_std',
 'WP_coif1_hdv_mean',
 'WP_coif1_hdv_std',
 'WP_coif1_hdd_mean',
 'WP_coif1_hdd_std',
 'WP_coif1_vaa_mean',
 'WP_coif1_vaa_std',
 'WP_coif1_vah_mean',
 'WP_coif1_vah_std',
 'WP_coif1_vav_mean',
 'WP_coif1_vav_std',
 'WP_coif1_vad_mean',
 'WP_coif1_vad_std',
 'WP_coif1_vha_mean',
 'WP_coif1_vha_std',
 'WP_coif1_vhh_mean',
 'WP_coif1_vhh_std',
 'WP_coif1_vhv_mean',
 'WP_coif1_vhv_std',
 'WP_coif1_vhd_mean',
 'WP_coif1_vhd_std',
 'WP_coif1_vva_mean',
 'WP_coif1_vva_std',
 'WP_coif1_vvh_mean',
 'WP_coif1_vvh_std',
 'WP_coif1_vvv_mean',
 'WP_coif1_vvv_std',
 'WP_coif1_vvd_mean',
 'WP_coif1_vvd_std',
 'WP_coif1_vda_mean',
 'WP_coif1_vda_std',
 'WP_coif1_vdh_mean',
 'WP_coif1_vdh_std',
 'WP_coif1_vdv_mean',
 'WP_coif1_vdv_std',
 'WP_coif1_vdd_mean',
 'WP_coif1_vdd_std',
 'WP_coif1_daa_mean',
 'WP_coif1_daa_std',
 'WP_coif1_dah_mean',
 'WP_coif1_dah_std',
 'WP_coif1_dav_mean',
 'WP_coif1_dav_std',
 'WP_coif1_dad_mean',
 'WP_coif1_dad_std',
 'WP_coif1_dha_mean',
 'WP_coif1_dha_std',
 'WP_coif1_dhh_mean',
 'WP_coif1_dhh_std',
 'WP_coif1_dhv_mean',
 'WP_coif1_dhv_std',
 'WP_coif1_dhd_mean',
 'WP_coif1_dhd_std',
 'WP_coif1_dva_mean',
 'WP_coif1_dva_std',
 'WP_coif1_dvh_mean',
 'WP_coif1_dvh_std',
 'WP_coif1_dvv_mean',
 'WP_coif1_dvv_std',
 'WP_coif1_dvd_mean',
 'WP_coif1_dvd_std',
 'WP_coif1_dda_mean',
 'WP_coif1_dda_std',
 'WP_coif1_ddh_mean',
 'WP_coif1_ddh_std',
 'WP_coif1_ddv_mean',
 'WP_coif1_ddv_std',
 'WP_coif1_ddd_mean',
 'WP_coif1_ddd_std']
dwpfc=pd.DataFrame(dwpf).T
#dgt1,dgt2=features['D_GT']
dgtf=['GT_th_0.0_freq_0.05_mean',
 'GT_th_0.0_freq_0.05_std',
 'GT_th_0.0_freq_0.4_mean',
 'GT_th_0.0_freq_0.4_std',
 'GT_th_1.0_freq_0.05_mean',
 'GT_th_1.0_freq_0.05_std',
 'GT_th_1.0_freq_0.4_mean',
 'GT_th_1.0_freq_0.4_std',
 'GT_th_2.0_freq_0.05_mean',
 'GT_th_2.0_freq_0.05_std',
 'GT_th_2.0_freq_0.4_mean',
 'GT_th_2.0_freq_0.4_std',
 'GT_th_3.0_freq_0.05_mean',
 'GT_th_3.0_freq_0.05_std',
 'GT_th_3.0_freq_0.4_mean',
 'GT_th_3.0_freq_0.4_std']

dgtfc=pd.DataFrame(dgtf).T
#dam1,dam2=features['D_AMFM']
damf=['AMFM_low0',
 'AMFM_low1',
 'AMFM_low2',
 'AMFM_low3',
 'AMFM_low4',
 'AMFM_low5',
 'AMFM_low6',
 'AMFM_low7',
 'AMFM_low8',
 'AMFM_low9',
 'AMFM_low10',
 'AMFM_low11',
 'AMFM_low12',
 'AMFM_low13',
 'AMFM_low14',
 'AMFM_low15',
 'AMFM_low16',
 'AMFM_low17',
 'AMFM_low18',
 'AMFM_low19',
 'AMFM_low20',
 'AMFM_low21',
 'AMFM_low22',
 'AMFM_low23',
 'AMFM_low24',
 'AMFM_low25',
 'AMFM_low26',
 'AMFM_low27',
 'AMFM_low28',
 'AMFM_low29',
 'AMFM_low30',
 'AMFM_low31',
 'AMFM_med0',
 'AMFM_med1',
 'AMFM_med2',
 'AMFM_med3',
 'AMFM_med4',
 'AMFM_med5',
 'AMFM_med6',
 'AMFM_med7',
 'AMFM_med8',
 'AMFM_med9',
 'AMFM_med10',
 'AMFM_med11',
 'AMFM_med12',
 'AMFM_med13',
 'AMFM_med14',
 'AMFM_med15',
 'AMFM_med16',
 'AMFM_med17',
 'AMFM_med18',
 'AMFM_med19',
 'AMFM_med20',
 'AMFM_med21',
 'AMFM_med22',
 'AMFM_med23',
 'AMFM_med24',
 'AMFM_med25',
 'AMFM_med26',
 'AMFM_med27',
 'AMFM_med28',
 'AMFM_med29',
 'AMFM_med30',
 'AMFM_med31',
 'AMFM_high0',
 'AMFM_high1',
 'AMFM_high2',
 'AMFM_high3',
 'AMFM_high4',
 'AMFM_high5',
 'AMFM_high6',
 'AMFM_high7',
 'AMFM_high8',
 'AMFM_high9',
 'AMFM_high10',
 'AMFM_high11',
 'AMFM_high12',
 'AMFM_high13',
 'AMFM_high14',
 'AMFM_high15',
 'AMFM_high16',
 'AMFM_high17',
 'AMFM_high18',
 'AMFM_high19',
 'AMFM_high20',
 'AMFM_high21',
 'AMFM_high22',
 'AMFM_high23',
 'AMFM_high24',
 'AMFM_high25',
 'AMFM_high26',
 'AMFM_high27',
 'AMFM_high28',
 'AMFM_high29',
 'AMFM_high30',
 'AMFM_high31',
 'AMFM_dc0',
 'AMFM_dc1',
 'AMFM_dc2',
 'AMFM_dc3',
 'AMFM_dc4',
 'AMFM_dc5',
 'AMFM_dc6',
 'AMFM_dc7',
 'AMFM_dc8',
 'AMFM_dc9',
 'AMFM_dc10',
 'AMFM_dc11',
 'AMFM_dc12',
 'AMFM_dc13',
 'AMFM_dc14',
 'AMFM_dc15',
 'AMFM_dc16',
 'AMFM_dc17',
 'AMFM_dc18',
 'AMFM_dc19',
 'AMFM_dc20',
 'AMFM_dc21',
 'AMFM_dc22',
 'AMFM_dc23',
 'AMFM_dc24',
 'AMFM_dc25',
 'AMFM_dc26',
 'AMFM_dc27',
 'AMFM_dc28',
 'AMFM_dc29',
 'AMFM_dc30',
 'AMFM_dc31']
# #ffreq
damfc=pd.DataFrame(damf).T
path_new='C:\\Users\\mirazzak\\knight_test_data_corrected\\images'
lsitd=os.listdir(path_new)
for i in lsitd:
    print(i)
    #pathm=os.path.join(path_new,i+'\imaging.nii.gz')
    pathm=os.path.join(path_new,i)
    print(pathm)
    read_img=sitk.ReadImage(pathm)
    #get_array_img=sitk.GetArrayFromImage(read_img)

    # convert to numpy
    data_npy = sitk.GetArrayFromImage(read_img).astype(np.float32)
    inner_image = kits_normalization(data_npy)
    out_im=preprocessing(inner_image)
    out_im=maxprojection(out_im)
    image=out_im
    #df8.loc[i,'subject_id']=i
    features1={}
    features2={}
    features3={}
    features4={}
    features5={}
    #features['A_FOS'] = fos(out_im,None)
    #df8.loc[i,:]=features['A_FOS'][0]
    featuresf={}
    #features['A_GLCM'] = glcm_features(out_im, ignore_zeros=True)
    #featuresf['C_Histogram'] = histogram(out_im, None, bins=32) # can be used for 3D
    #features['D_DWT'] = dwt_features(image,None, wavelet='bior3.3', levels=3)
    #features['D_SWT'] = swt_features(image,None, wavelet='bior3.3', levels=3)
    #dfc.loc[i,:]=features['A_GLCM'][0]
    #dfd.loc[i,:]=features['A_GLCM'][1]
    features1['D_DWT'] = dwt_features(image,None, wavelet='bior3.3', levels=3)
    d2ff.loc[i,:]=features1['D_DWT'][0]
    features2['D_SWT'] = swt_features(image,None, wavelet='bior3.3', levels=3)
    ds2ff.loc[i,:]=features2['D_SWT'][0]
    features3['D_WP'] = wp_features(image, None, wavelet='coif1', maxlevel=3)
    dwpfc.loc[i,:]=features3['D_WP'][0]
    features4['D_GT'] = gt_features(image, None)
    dgtfc.loc[i,:]=features4['D_GT'][0]
    #features5['D_AMFM'] = amfm_features(image)
    #damfc[i,:]=features5['D_AMFM'][0]
    #break
#%%
testdf=d2ff

firstdf=testdf.reset_index()
firstdf=firstdf.drop(0)
old_columns=(firstdf.columns)
#new_list=d2f
d2f=['SubjectId','DWT_bior3.3_level_1_da_mean',
 'DWT_bior3.3_level_1_da_std',
 'DWT_bior3.3_level_1_dd_mean',
 'DWT_bior3.3_level_1_dd_std',
 'DWT_bior3.3_level_1_ad_mean',
 'DWT_bior3.3_level_1_ad_std',
 'DWT_bior3.3_level_2_da_mean',
 'DWT_bior3.3_level_2_da_std',
 'DWT_bior3.3_level_2_dd_mean',
 'DWT_bior3.3_level_2_dd_std',
 'DWT_bior3.3_level_2_ad_mean',
 'DWT_bior3.3_level_2_ad_std',
 'DWT_bior3.3_level_3_da_mean',
 'DWT_bior3.3_level_3_da_std',
 'DWT_bior3.3_level_3_dd_mean',
 'DWT_bior3.3_level_3_dd_std',
 'DWT_bior3.3_level_3_ad_mean',
 'DWT_bior3.3_level_3_ad_std']
new_list=d2f

firstdf.rename(columns={old_columns[idx]: name for (idx,name) in enumerate(new_list)}, inplace=True)
firstdf.to_csv('D_DWT_featurestest.csv',index=False)
#%%
#ds2ff=pd.DataFrame(ds2f).T
testdf=dwpfc.copy()
seconddf=testdf.reset_index()
seconddf=seconddf.drop(0)
old_columns=(seconddf.columns)
#new_list=d2f
#new_list=ds2f
dwpf=['SubjectId','WP_coif1_aah_mean',
 'WP_coif1_aah_std',
 'WP_coif1_aav_mean',
 'WP_coif1_aav_std',
 'WP_coif1_aad_mean',
 'WP_coif1_aad_std',
 'WP_coif1_aha_mean',
 'WP_coif1_aha_std',
 'WP_coif1_ahh_mean',
 'WP_coif1_ahh_std',
 'WP_coif1_ahv_mean',
 'WP_coif1_ahv_std',
 'WP_coif1_ahd_mean',
 'WP_coif1_ahd_std',
 'WP_coif1_ava_mean',
 'WP_coif1_ava_std',
 'WP_coif1_avh_mean',
 'WP_coif1_avh_std',
 'WP_coif1_avv_mean',
 'WP_coif1_avv_std',
 'WP_coif1_avd_mean',
 'WP_coif1_avd_std',
 'WP_coif1_ada_mean',
 'WP_coif1_ada_std',
 'WP_coif1_adh_mean',
 'WP_coif1_adh_std',
 'WP_coif1_adv_mean',
 'WP_coif1_adv_std',
 'WP_coif1_add_mean',
 'WP_coif1_add_std',
 'WP_coif1_haa_mean',
 'WP_coif1_haa_std',
 'WP_coif1_hah_mean',
 'WP_coif1_hah_std',
 'WP_coif1_hav_mean',
 'WP_coif1_hav_std',
 'WP_coif1_had_mean',
 'WP_coif1_had_std',
 'WP_coif1_hha_mean',
 'WP_coif1_hha_std',
 'WP_coif1_hhh_mean',
 'WP_coif1_hhh_std',
 'WP_coif1_hhv_mean',
 'WP_coif1_hhv_std',
 'WP_coif1_hhd_mean',
 'WP_coif1_hhd_std',
 'WP_coif1_hva_mean',
 'WP_coif1_hva_std',
 'WP_coif1_hvh_mean',
 'WP_coif1_hvh_std',
 'WP_coif1_hvv_mean',
 'WP_coif1_hvv_std',
 'WP_coif1_hvd_mean',
 'WP_coif1_hvd_std',
 'WP_coif1_hda_mean',
 'WP_coif1_hda_std',
 'WP_coif1_hdh_mean',
 'WP_coif1_hdh_std',
 'WP_coif1_hdv_mean',
 'WP_coif1_hdv_std',
 'WP_coif1_hdd_mean',
 'WP_coif1_hdd_std',
 'WP_coif1_vaa_mean',
 'WP_coif1_vaa_std',
 'WP_coif1_vah_mean',
 'WP_coif1_vah_std',
 'WP_coif1_vav_mean',
 'WP_coif1_vav_std',
 'WP_coif1_vad_mean',
 'WP_coif1_vad_std',
 'WP_coif1_vha_mean',
 'WP_coif1_vha_std',
 'WP_coif1_vhh_mean',
 'WP_coif1_vhh_std',
 'WP_coif1_vhv_mean',
 'WP_coif1_vhv_std',
 'WP_coif1_vhd_mean',
 'WP_coif1_vhd_std',
 'WP_coif1_vva_mean',
 'WP_coif1_vva_std',
 'WP_coif1_vvh_mean',
 'WP_coif1_vvh_std',
 'WP_coif1_vvv_mean',
 'WP_coif1_vvv_std',
 'WP_coif1_vvd_mean',
 'WP_coif1_vvd_std',
 'WP_coif1_vda_mean',
 'WP_coif1_vda_std',
 'WP_coif1_vdh_mean',
 'WP_coif1_vdh_std',
 'WP_coif1_vdv_mean',
 'WP_coif1_vdv_std',
 'WP_coif1_vdd_mean',
 'WP_coif1_vdd_std',
 'WP_coif1_daa_mean',
 'WP_coif1_daa_std',
 'WP_coif1_dah_mean',
 'WP_coif1_dah_std',
 'WP_coif1_dav_mean',
 'WP_coif1_dav_std',
 'WP_coif1_dad_mean',
 'WP_coif1_dad_std',
 'WP_coif1_dha_mean',
 'WP_coif1_dha_std',
 'WP_coif1_dhh_mean',
 'WP_coif1_dhh_std',
 'WP_coif1_dhv_mean',
 'WP_coif1_dhv_std',
 'WP_coif1_dhd_mean',
 'WP_coif1_dhd_std',
 'WP_coif1_dva_mean',
 'WP_coif1_dva_std',
 'WP_coif1_dvh_mean',
 'WP_coif1_dvh_std',
 'WP_coif1_dvv_mean',
 'WP_coif1_dvv_std',
 'WP_coif1_dvd_mean',
 'WP_coif1_dvd_std',
 'WP_coif1_dda_mean',
 'WP_coif1_dda_std',
 'WP_coif1_ddh_mean',
 'WP_coif1_ddh_std',
 'WP_coif1_ddv_mean',
 'WP_coif1_ddv_std',
 'WP_coif1_ddd_mean',
 'WP_coif1_ddd_std']
new_list=dwpf
seconddf.rename(columns={old_columns[idx]: name for (idx,name) in enumerate(new_list)}, inplace=True)
seconddf.to_csv('D_WP_features_test.csv',index=False)


#%%
dfs=pd.DataFrame(lsitd,columns=['SubjectId'])
dff=d2ff
firstdf=dff.drop(0)
firstdff=firstdf.rename(columns={'Index':'SubjectId'})
firstdf=firstdf.reset_index()
firstdff=firstdf.rename(columns={'index':'SubjectId'})
old_columns=(firstdff.columns)
#df.rename(columns={})
for i, df in enumerate(d2f,1):
    print(df)
    firstdff.columns=[col_name+'{}'.format(i) for col_name in firstdff.columns]
firstdff.columns=[i for i in firstdff.columns]
new_list=d2f
dff.rename(columns={old_columns[idx]: name for (idx,name) in enumerate(new_list)}, inplace=True)

results1=pd.concat([df8,df2],axis=0,ignore_index=True)
df=df8.merge(df2,how='left',left_on=0,right_on=0)
df3=df8.rename()
df8.index
df8.rename(index={'subject':'0'},inplace=True)
df8.index=Features_name=['Sub','FOS_Mean',
 'FOS_Variance',
 'FOS_Median',
 'FOS_Mode',
 'FOS_Skewness',
 'FOS_Kurtosis',
 'FOS_Energy',
 'FOS_Entropy',
 'FOS_MinimalGrayLevel',
 'FOS_MaximalGrayLevel',
 'FOS_CoefficientOfVariation',
 'FOS_10Percentile',
 'FOS_25Percentile',
 'FOS_75Percentile',
 'FOS_90Percentile',
 'FOS_HistogramWidth'].T
df8.columns
dd=df8.reset_index()
dd1=dd.drop(0)
dd2=dd1.reset_index()
dd3=dd1.rename(columns={'index':'SubjectId'})
dd3.to_csv('fos_features1.csv',index=False)


#%% E. Other
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pyfeats import *
path='C:\\Users\\mirazzak\\data\\case_00000'

pathf=os.path.join(path,'imaging.nii.gz')

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




# read_img=sitk.ReadImage(pathf)
# #get_array_img=sitk.GetArrayFromImage(read_img)

# # convert to numpy
# data_npy = sitk.GetArrayFromImage(read_img).astype(np.float32)
# inner_image = kits_normalization(data_npy)
# out_im=preprocessing(inner_image)
# out_im1=np.swapaxes(out_im,2,0)

def maxprojection(img):
    ind=img.argmax(axis=0)
    a1,a2=np.indices(ind.shape)
    maxp=img[ind,a1,a2]
    return maxp

#plt.imshow(out_im1[3,:,:])
#maximg=maxprojection(out_im1)
# import matplotlib.pyplot as plt
# plt.imshow(maximg)

# image=out_im
# image=maximg

# features = {}
# features['A_FOS'] = fos(image,None)

import pandas as pd
#data1=pd.DataFrame(y)
path_new='C:\\Users\\data2\\data'
import pandas as pd
#data1=pd.DataFrame(y)
path_new='C:\\Users\\data2\\data'

lsitd=os.listdir(path_new)

lsitd=os.listdir(path_new)
features['E_HOG'] = hog_features(image, ppc=8, cpb=3)
features['E_HuMoments'] = hu_moments(image)
#features['E_TAS'] = tas_features(image.astype(int16))
features['E_ZernikesMoments'] = zernikes_moments(image, radius=9)

damfc=pd.DataFrame(damf).T

for i in lsitd:
    print(i)
    pathm=os.path.join(path_new,i+'\imaging.nii.gz')
    print(pathm)
    read_img=sitk.ReadImage(pathm)
    #get_array_img=sitk.GetArrayFromImage(read_img)

    # convert to numpy
    data_npy = sitk.GetArrayFromImage(read_img).astype(np.float32)
    inner_image = kits_normalization(data_npy)
    out_im=preprocessing(inner_image)
    out_im=maxprojection(out_im)
    image=out_im
    #df8.loc[i,'subject_id']=i
    features1={}
    features2={}
    features3={}
    features4={}
    features5={}
    #features['A_FOS'] = fos(out_im,None)
    #df8.loc[i,:]=features['A_FOS'][0]
    featuresf={}
    #features['A_GLCM'] = glcm_features(out_im, ignore_zeros=True)
    #featuresf['C_Histogram'] = histogram(out_im, None, bins=32) # can be used for 3D
    #features['D_DWT'] = dwt_features(image,None, wavelet='bior3.3', levels=3)
    #features['D_SWT'] = swt_features(image,None, wavelet='bior3.3', levels=3)
    #dfc.loc[i,:]=features['A_GLCM'][0]
    #dfd.loc[i,:]=features['A_GLCM'][1]
    # features1['D_DWT'] = dwt_features(image,None, wavelet='bior3.3', levels=3)
    # d2ff.loc[i,:]=features1['D_DWT'][0]
    # features2['D_SWT'] = swt_features(image,None, wavelet='bior3.3', levels=3)
    # ds2ff.loc[i,:]=features2['D_SWT'][0]
    # features3['D_WP'] = wp_features(image, None, wavelet='coif1', maxlevel=3)
    # dwpfc.loc[i,:]=features3['D_WP'][0]
    # features4['D_GT'] = gt_features(image, None)
    # dgtfc.loc[i,:]=features4['D_GT'][0]
    features1['E_HOG'] = hog_features(image, ppc=8, cpb=3)
    features2['E_HuMoments'] = hu_moments(image)
    #features['E_TAS'] = tas_features(image.astype(int16))
    features3['E_ZernikesMoments'] = zernikes_moments(image, radius=9)
    #break











