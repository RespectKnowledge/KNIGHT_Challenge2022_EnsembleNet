# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 18:28:31 2022

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
        
        self.feature=self.in_out.drop(['SubjectId','task_1_label','task_2_label'],axis=1, inplace=True)

        
        self.path_img=[]
        
        for i in self.ids:
            pth_i=os.path.join(self.images,i)
            pth_ie=os.path.join(pth_i+'/imaging.nii.gz')
            self.path_img.append(pth_ie)
            
        
    def __getitem__(self,idx):
        
        feat=self.in_out.iloc[idx]
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
            
        if inner_image_height != 512 or inner_image_width != 512:
            # there is one sample in KiTS21 with a 796 number of rows,
            # different from all the others. we disregard it for simplicity
            return None
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
    
    

filet=pd.read_csv('/home/imranr/knights/trainfold0.csv')
data='/home/imranr/knights/data'

filev=pd.read_csv('/home/imranr/knights/validfold0.csv')

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

i,f, o=valid_dataset[2]

print(i.shape)
print(i.min())
print(i.max())
print(o)
print(f)

# i,f, o=train_dataset[2]

# print(i.shape)
# print(i.min())
# print(i.max())
# print(o)
# print(f)


#%
from torch.utils.data import DataLoader

# train_dataloader = DataLoader(ob_d, batch_size=4, shuffle=True)
# #test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
# for i, data in enumerate(train_dataloader):
#     img,f,l=data
#     print(img.shape)
#     print(f.shape)
#     print(l)
#     break
    

batch=4

train_dataloader = DataLoader(train_dataset, batch_size= batch,pin_memory=True,num_workers=6, shuffle=True)

valid_dataloader = DataLoader(valid_dataset, batch_size= batch,pin_memory=True,num_workers=6, shuffle=False)


import sys 
import os
import glob
import time
import random
import os
import glob
import time
import random
import torch
import numpy as np
import pandas as pd
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from collections import OrderedDict
from typing import Callable, Sequence, Type, Union

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

from monai.networks.layers.factories import Conv, Dropout, Pool
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.utils.module import look_up_option

__all__ = [
    "DenseNet",
    "Densenet",
    "DenseNet121",
    "densenet121",
    "Densenet121",
    "DenseNet169",
    "densenet169",
    "Densenet169",
    "DenseNet201",
    "densenet201",
    "Densenet201",
    "DenseNet264",
    "densenet264",
    "Densenet264",
]


class _DenseLayer(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        growth_rate: int,
        bn_size: int,
        dropout_prob: float,
        act: Union[str, tuple] = ("relu", {"inplace": True}),
        norm: Union[str, tuple] = "batch",
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions of the input image.
            in_channels: number of the input channel.
            growth_rate: how many filters to add each layer (k in paper).
            bn_size: multiplicative factor for number of bottle neck layers.
                (i.e. bn_size * k features in the bottleneck layer)
            dropout_prob: dropout rate after each dense layer.
            act: activation type and arguments. Defaults to relu.
            norm: feature normalization type and arguments. Defaults to batch norm.
        """
        super().__init__()

        out_channels = bn_size * growth_rate
        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        dropout_type: Callable = Dropout[Dropout.DROPOUT, spatial_dims]

        self.layers = nn.Sequential()

        self.layers.add_module("norm1", get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels))
        self.layers.add_module("relu1", get_act_layer(name=act))
        self.layers.add_module("conv1", conv_type(in_channels, out_channels, kernel_size=1, bias=False))

        self.layers.add_module("norm2", get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=out_channels))
        self.layers.add_module("relu2", get_act_layer(name=act))
        self.layers.add_module("conv2", conv_type(out_channels, growth_rate, kernel_size=3, padding=1, bias=False))

        if dropout_prob > 0:
            self.layers.add_module("dropout", dropout_type(dropout_prob))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        new_features = self.layers(x)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(
        self,
        spatial_dims: int,
        layers: int,
        in_channels: int,
        bn_size: int,
        growth_rate: int,
        dropout_prob: float,
        act: Union[str, tuple] = ("relu", {"inplace": True}),
        norm: Union[str, tuple] = "batch",
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions of the input image.
            layers: number of layers in the block.
            in_channels: number of the input channel.
            bn_size: multiplicative factor for number of bottle neck layers.
                (i.e. bn_size * k features in the bottleneck layer)
            growth_rate: how many filters to add each layer (k in paper).
            dropout_prob: dropout rate after each dense layer.
            act: activation type and arguments. Defaults to relu.
            norm: feature normalization type and arguments. Defaults to batch norm.
        """
        super().__init__()
        for i in range(layers):
            layer = _DenseLayer(spatial_dims, in_channels, growth_rate, bn_size, dropout_prob, act=act, norm=norm)
            in_channels += growth_rate
            self.add_module("denselayer%d" % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        act: Union[str, tuple] = ("relu", {"inplace": True}),
        norm: Union[str, tuple] = "batch",
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions of the input image.
            in_channels: number of the input channel.
            out_channels: number of the output classes.
            act: activation type and arguments. Defaults to relu.
            norm: feature normalization type and arguments. Defaults to batch norm.
        """
        super().__init__()

        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        pool_type: Callable = Pool[Pool.AVG, spatial_dims]

        self.add_module("norm", get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels))
        self.add_module("relu", get_act_layer(name=act))
        self.add_module("conv", conv_type(in_channels, out_channels, kernel_size=1, bias=False))
        self.add_module("pool", pool_type(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    """
    Densenet based on: `Densely Connected Convolutional Networks <https://arxiv.org/pdf/1608.06993.pdf>`_.
    Adapted from PyTorch Hub 2D version: https://pytorch.org/vision/stable/models.html#id16.
    This network is non-determistic When `spatial_dims` is 3 and CUDA is enabled. Please check the link below
    for more details:
    https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms
    Args:
        spatial_dims: number of spatial dimensions of the input image.
        in_channels: number of the input channel.
        out_channels: number of the output classes.
        init_features: number of filters in the first convolution layer.
        growth_rate: how many filters to add each layer (k in paper).
        block_config: how many layers in each pooling block.
        bn_size: multiplicative factor for number of bottle neck layers.
            (i.e. bn_size * k features in the bottleneck layer)
        act: activation type and arguments. Defaults to relu.
        norm: feature normalization type and arguments. Defaults to batch norm.
        dropout_prob: dropout rate after each dense layer.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        init_features: int = 64,
        growth_rate: int = 32,
        block_config: Sequence[int] = (6, 12, 24, 16),
        bn_size: int = 4,
        act: Union[str, tuple] = ("relu", {"inplace": True}),
        norm: Union[str, tuple] = "batch",
        dropout_prob: float = 0.0,
    ) -> None:

        super().__init__()

        conv_type: Type[Union[nn.Conv1d, nn.Conv2d, nn.Conv3d]] = Conv[Conv.CONV, spatial_dims]
        pool_type: Type[Union[nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d]] = Pool[Pool.MAX, spatial_dims]
        avg_pool_type: Type[Union[nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d]] = Pool[
            Pool.ADAPTIVEAVG, spatial_dims
        ]

        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("conv0", conv_type(in_channels, init_features, kernel_size=7, stride=2, padding=3, bias=False)),
                    ("norm0", get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=init_features)),
                    ("relu0", get_act_layer(name=act)),
                    ("pool0", pool_type(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )

        in_channels = init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                spatial_dims=spatial_dims,
                layers=num_layers,
                in_channels=in_channels,
                bn_size=bn_size,
                growth_rate=growth_rate,
                dropout_prob=dropout_prob,
                act=act,
                norm=norm,
            )
            self.features.add_module(f"denseblock{i + 1}", block)
            in_channels += num_layers * growth_rate
            if i == len(block_config) - 1:
                self.features.add_module(
                    "norm5", get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
                )
            else:
                _out_channels = in_channels // 2
                trans = _Transition(
                    spatial_dims, in_channels=in_channels, out_channels=_out_channels, act=act, norm=norm
                )
                self.features.add_module(f"transition{i + 1}", trans)
                in_channels = _out_channels

        # pooling and classification
        self.class_layers = nn.Sequential(
            OrderedDict(
                [
                    ("relu", get_act_layer(name=act)),
                    ("pool", avg_pool_type(1)),
                    ("flatten", nn.Flatten(1)),
                    ("out", nn.Linear(in_channels, out_channels)),
                ]
            )
        )

        self.feature_layers=nn.Sequential(get_act_layer(name=act),avg_pool_type(1),
                                          nn.Flatten(1))

        for m in self.modules():
            if isinstance(m, conv_type):
                nn.init.kaiming_normal_(torch.as_tensor(m.weight))
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(torch.as_tensor(m.weight), 1)
                nn.init.constant_(torch.as_tensor(m.bias), 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(torch.as_tensor(m.bias), 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x1=self.feature_layers(x)
        x = self.class_layers(x)
        return x,x1


def _load_state_dict(model: nn.Module, arch: str, progress: bool):
    """
    This function is used to load pretrained models.
    Adapted from PyTorch Hub 2D version: https://pytorch.org/vision/stable/models.html#id16.
    """
    model_urls = {
        "densenet121": "https://download.pytorch.org/models/densenet121-a639ec97.pth",
        "densenet169": "https://download.pytorch.org/models/densenet169-b2777c0a.pth",
        "densenet201": "https://download.pytorch.org/models/densenet201-c1103571.pth",
    }
    model_url = look_up_option(arch, model_urls, None)
    if model_url is None:
        raise ValueError(
            "only 'densenet121', 'densenet169' and 'densenet201' are supported to load pretrained weights."
        )

    pattern = re.compile(
        r"^(.*denselayer\d+)(\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$"
    )

    state_dict = load_state_dict_from_url(model_url, progress=progress)
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + ".layers" + res.group(2) + res.group(3)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]

    model_dict = model.state_dict()
    state_dict = {
        k: v for k, v in state_dict.items() if (k in model_dict) and (model_dict[k].shape == state_dict[k].shape)
    }
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)


class DenseNet121(DenseNet):
    """DenseNet121 with optional pretrained support when `spatial_dims` is 2."""

    def __init__(
        self,
        init_features: int = 64,
        growth_rate: int = 32,
        block_config: Sequence[int] = (6, 12, 24, 16),
        pretrained: bool = False,
        progress: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(init_features=init_features, growth_rate=growth_rate, block_config=block_config, **kwargs)
        if pretrained:
            if kwargs["spatial_dims"] > 2:
                raise NotImplementedError(
                    "Parameter `spatial_dims` is > 2 ; currently PyTorch Hub does not"
                    "provide pretrained models for more than two spatial dimensions."
                )
            _load_state_dict(self, "densenet121", progress)


class DenseNet169(DenseNet):
    """DenseNet169 with optional pretrained support when `spatial_dims` is 2."""

    def __init__(
        self,
        init_features: int = 64,
        growth_rate: int = 32,
        block_config: Sequence[int] = (6, 12, 32, 32),
        pretrained: bool = False,
        progress: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(init_features=init_features, growth_rate=growth_rate, block_config=block_config, **kwargs)
        if pretrained:
            if kwargs["spatial_dims"] > 2:
                raise NotImplementedError(
                    "Parameter `spatial_dims` is > 2 ; currently PyTorch Hub does not"
                    "provide pretrained models for more than two spatial dimensions."
                )
            _load_state_dict(self, "densenet169", progress)


class DenseNet201(DenseNet):
    """DenseNet201 with optional pretrained support when `spatial_dims` is 2."""

    def __init__(
        self,
        init_features: int = 64,
        growth_rate: int = 32,
        block_config: Sequence[int] = (6, 12, 48, 32),
        pretrained: bool = False,
        progress: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(init_features=init_features, growth_rate=growth_rate, block_config=block_config, **kwargs)
        if pretrained:
            if kwargs["spatial_dims"] > 2:
                raise NotImplementedError(
                    "Parameter `spatial_dims` is > 2 ; currently PyTorch Hub does not"
                    "provide pretrained models for more than two spatial dimensions."
                )
            _load_state_dict(self, "densenet201", progress)


class DenseNet264(DenseNet):
    """DenseNet264"""

    def __init__(
        self,
        init_features: int = 64,
        growth_rate: int = 32,
        block_config: Sequence[int] = (6, 12, 64, 48),
        pretrained: bool = False,
        progress: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(init_features=init_features, growth_rate=growth_rate, block_config=block_config, **kwargs)
        if pretrained:
            raise NotImplementedError("Currently PyTorch Hub does not provide densenet264 pretrained models.")


Densenet = DenseNet
Densenet121 = densenet121 = DenseNet121
Densenet169 = densenet169 = DenseNet169
Densenet201 = densenet201 = DenseNet201
Densenet264 = densenet264 = DenseNet264



import torch.nn as nn
class classify_layer(nn.Module):
    def __init__(self,in_features,num_classes):
        super(classify_layer,self).__init__()
        self.classifier=nn.Sequential(nn.Linear(in_features,768),
                                      nn.ReLU(True),
                                      nn.Linear(768,num_classes))
        print(self.classifier)
        
    def forward(self,x):
        x=self.classifier(x)
        return x
class fusemodel(nn.Module):
  def __init__(self,n_classes):
    super(fusemodel,self).__init__()
    self.model=DenseNet121(spatial_dims=3, in_channels=1,out_channels=2)
    #self.classifier=nn.Sequential()
    #self.feature_layers=nn.Sequential(get_act_layer(name=act),avg_pool_type(1),
                                          #nn.Flatten(1))
    self.fc=nn.Sequential(nn.Linear(1024,512),
                              nn.Dropout(0.3),
                              nn.ReLU(True),
                              )
    self.classifier=nn.Linear(1034,n_classes)
  def forward(self,x,x1):
    out1,features=self.model(x)
    #Concat1=torch.cat((features,x1),dim=1) # 768+768=512
    #featuresfc=self.fc(features)   # shape 512
    Concat1=torch.cat((features,x1),dim=1) # 512+10=522
    out=self.classifier(Concat1)
    return out

import torch
#inp=torch.rand(2,3,32,64,64)
#inp2=torch.rand(2,10)
model=fusemodel(2)
#out1=model(inp,inp2)
#print(out1.shape)

model=nn.DataParallel(model)
model=model.to(device)
#loss_func=nn.BCEWithLogitsLoss()
from tqdm import tqdm
import torch.optim as optim
optimizer=optim.Adam(model.parameters(),lr=0.0001)

#classes=['NRG','RG']
classes=['NoAT','CanAT']
#c1=data['label'].value_counts()[0]
#c2=data['label'].value_counts()[1]
my_distribution=np.array([175,65])
class_weights = torch.from_numpy(np.divide(1, my_distribution)).float().to(device)
class_weights = class_weights / class_weights.sum()
for i, c in enumerate(classes):
  print('Weight for class %s: %f' % (c, class_weights.cpu().numpy()[i]))
loss_func = nn.CrossEntropyLoss(weight=class_weights)
#label=label.to(torch.int64)
################################ training functions ###################
def train_fn(model,train_loader):
    model.train()
    counter=0
    training_run_loss=0.0
    train_running_correct=0.0
    for i, data in tqdm(enumerate(train_loader),total=int(len(train_dataset)/train_loader.batch_size)):
        counter+=1
        # extract dataset
        imge,feature,label=data
        imge=imge.float()
        #label=label.float()
        label.to(torch.int64)
        feature=feature.float()
        feature=feature.to(device)
        imge=imge.to(device)
        label=label.to(device)
        #imge=imge.cuda()
        #label=label.cuda()
        # zero_out the gradient
        optimizer.zero_grad()
        output=model(imge,feature)
        loss=loss_func(output,label)
        training_run_loss+=loss.item()
        _,preds=torch.max(output.data,1)
        train_running_correct+=(preds==label).sum().item()
        loss.backward()
        optimizer.step()
    ###################### state computation ###################
    train_loss=training_run_loss/len(train_loader.dataset)
    train_loss_ep.append(train_loss)
    train_accuracy=100.* train_running_correct/len(train_loader.dataset)
    train_accuracy_ep.append(train_accuracy)
    print(f"Train Loss:{train_loss:.4f}, Train Acc:{train_accuracy:0.2f}")
    return train_loss_ep,train_accuracy_ep

########################## validation function ##################
def validation_fn(model,valid_loader):
  # evluation start
    print("validation start")
    
    model.eval()
    val_running_loss = 0.0
    val_running_correct = 0
    with torch.no_grad():
        for i,data in tqdm(enumerate(valid_loader),total=int(len(valid_loader)/valid_loader.batch_size)):
            imge,feature,label=data
            imge=imge.float()
            #label=label.float()
            label.to(torch.int64)
            feature=feature.float()
            feature=feature.to(device)
            imge=imge.to(device)
            label=label.to(device)
            #imge=imge.cuda()
            #label=label.cuda()
            output=model(imge,feature)
            loss=loss_func(output,label)
            val_running_loss+=loss.item()
            _,pred=torch.max(output.data,1)
            val_running_correct+=(pred==label).sum().item()
        val_loss=val_running_loss/len(valid_loader.dataset)
        val_loss_ep.append(val_loss)
        val_accuracy=100.* val_running_correct/(len(valid_loader.dataset))
        val_accuracy_ep.append(val_accuracy)
        print(f"Val Loss:{val_loss:0.4f}, Val_Acc:{val_accuracy:0.2f}")
        return val_loss_ep,val_accuracy_ep

import torch.optim as optim
optimizer=optim.Adam(model.parameters(),lr=0.0001)
train_loss_ep=[]
train_accuracy_ep=[]
val_loss_ep=[]
val_accuracy_ep=[]
lr = 3e-4
log = pd.DataFrame(index=[], columns=['epoch', 'lr', 'loss', 'accu', 'val_loss', 'val_accu'])
early_stop=20
epochs=500
best_acc = 0
name='3dcnnknighttask1'
trigger = 0
for epoch in range(epochs):
    print('Epoch [%d/%d]' %(epoch, epochs))
    # train for one epoch
    train_loss_ep,train_accuracy_ep=train_fn(model,train_dataloader)
    train_loss_ep1=np.mean(train_loss_ep)
    train_accuracy_ep1=np.mean(train_accuracy_ep)
    #y_pred,labels=Prediciton_fn(model,valid_loader)

    val_loss_ep,val_accuracy_ep=validation_fn(model,valid_dataloader)
    val_loss_ep1=np.mean(val_loss_ep)
    val_accuracy_ep1=np.mean(val_accuracy_ep)
    
    print('loss %.4f - accu %.4f - val_loss %.4f - val_accu %.4f'%(train_loss_ep1, train_accuracy_ep1, val_loss_ep1, val_accuracy_ep1))

    tmp = pd.Series([epoch,lr,train_loss_ep1,train_accuracy_ep1,val_loss_ep1,val_accuracy_ep1], index=['epoch', 'lr', 'loss', 'accu', 'val_loss', 'val_accu'])

    log = log.append(tmp, ignore_index=True)
    log.to_csv('models/%s/log.csv' %name, index=False)

    trigger += 1

    if val_accuracy_ep1 > best_acc:
        torch.save(model.state_dict(), 'models/%s/3dmodeltask1.pth' %name)
        best_acc = val_accuracy_ep1
        print("=> saved best model")
        trigger = 0

    # early stopping
    if not early_stop is None:
        if trigger >= early_stop:
            print("=> early stopping")
            break

    torch.cuda.empty_cache() 
