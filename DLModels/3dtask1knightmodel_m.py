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
        
        #feat=self.in_out.iloc[idx]
        #x_feature=pd.DataFrame(feat).T
        
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
        #x_f_array=np.array(x_feature)
        # x_f_array_t=torch.from_numpy(x_f_array).float()
        # x_f_array_t=torch.squeeze(x_f_array_t,axis=0)
        return (sample,y_output)
    
    def __len__(self):
        return(len(self.label))
    


#input_image = center_crop(input_image, new_shape=new_shape) 
    
    

filet=pd.read_csv('/home/imranr/knights/trainfold0_unnormalized.csv')
data='/home/imranr/knights/data'

filev=pd.read_csv('/home/imranr/knights/validfold0_unnormalized.csv')

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

i,o=valid_dataset[2]

print(i.shape)
print(i.min())
print(i.max())
print(o)
#print(f)

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

"""
(C) Copyright 2021 IBM Corp.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
Created on June 30, 2021
"""

from typing import Tuple, Any

import torch.nn as nn
from torch import Tensor
from torch.hub import load_state_dict_from_url
from torchvision.models.video.resnet import VideoResNet, BasicBlock, Conv3DSimple, BasicStem, model_urls


class FuseBackboneResnet3D(VideoResNet):
    """
    3D model classifier (ResNet architecture"
    """

    def __init__(self, pretrained: bool = False, in_channels: int = 1, name: str = "r3d_18") -> None:
        """
        Create 3D ResNet model
        :param pretrained: Use pretrained weights
        :param in_channels: number of input channels
        :param name: model name. currently only 'r3d_18' is supported
        """
        # init parameters per required backbone
        init_parameters = {
            'r3d_18': {'block': BasicBlock,
                       'conv_makers': [Conv3DSimple] * 4,
                       'layers': [2, 2, 2, 2],
                       'stem': BasicStem},
        }[name]
        # init original model
        super().__init__(**init_parameters)

        # load pretrained parameters if required
        if pretrained:
            state_dict = load_state_dict_from_url(model_urls[name])
            self.load_state_dict(state_dict)

        # save input parameters
        self.pretrained = pretrained
        self.in_channels = in_channels
        # override the first convolution layer to support any number of input channels
        self.stem = nn.Sequential(
            nn.Conv3d(self.in_channels, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2),
                      padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )

    def features(self, x: Tensor) -> Any:
        """
        Extract spatial features - given a 3D tensor
        :param x: Input tensor - shape: [batch_size, channels, z, y, x]
        :return: spatial features - shape [batch_size, n_features, z', y', x']
        """
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x: Tensor) -> Tuple[Tensor, None, None, None]:  # type: ignore
        """
        Forward pass. 3D global classification given a volume
        :param x: Input volume. shape: [batch_size, channels, z, y, x]
        :return: logits for global classification. shape: [batch_size, n_classes].
        """
        x = self.features(x)
        return x
    
backbone=FuseBackboneResnet3D(pretrained=True)   
from typing import Optional, Sequence
import torch.nn as nn

class ClassifierMLP(nn.Module):
    def __init__(self, in_ch: int, num_classes: Optional[int], layers_description: Sequence[int]=(256,128), dropout_rate: float = 0.0):
        super().__init__()
        layer_list = []
        layer_list.append(nn.Linear(in_ch, layers_description[0]))
        layer_list.append(nn.ReLU())
        if dropout_rate is not None and dropout_rate > 0:
            layer_list.append(nn.Dropout(p=dropout_rate))
        last_layer_size = layers_description[0]
        for curr_layer_size in layers_description[1:]:
            layer_list.append(nn.Linear(last_layer_size, curr_layer_size))
            layer_list.append(nn.ReLU())
            if dropout_rate is not None and dropout_rate > 0:
                layer_list.append(nn.Dropout(p=dropout_rate))
            last_layer_size = curr_layer_size
        
        if num_classes is not None:
            layer_list.append(nn.Linear(last_layer_size, num_classes))
        
        self.classifier = nn.Sequential(*layer_list)

    def forward(self, x):
        x = self.classifier(x)
        return x
  
# mlp=ClassifierMLP(10,None)  
# import torch    


# import torch    
# class FuseModelDefault(torch.nn.Module):
#     """
#     Default Fuse model - convolutional neural network with multiple heads
#     """

#     def __init__(self,fused_dropout_rate,num_classes,dropout_rate):
#         """
#         Default Fuse model - convolutional neural network with multiple heads
#         :param conv_inputs:     batch_dict name for convolutional backbone model input and its number of input channels. Unused if None. Kept for backward compatibility
#         :param backbone_args:   batch_dict name for generic backbone model input and its number of input channels. Unused if None
#         :param backbone:        PyTorch backbone module - a convolutional (in which case conv_inputs must be supplied) or some other (in which case backbone_args must be supplied) neural network
#         :param heads:           Sequence of head modules
#         """
#         super().__init__()
#         self.backbone = backbone
#         self.fused_dropout_rate=fused_dropout_rate
#         self.mlp=mlp
#         self.dropout_rate=dropout_rate
#         self.num_classes=num_classes
#         self.conv_classifier_3d = nn.Sequential(nn.Conv3d(640, 256, kernel_size=1),
#                                           nn.ReLU(),
#                                           nn.Dropout3d(p=fused_dropout_rate), 
#                                           nn.Conv3d(256, self.num_classes, kernel_size=1),
#                                           )
#         self.gmp = nn.AdaptiveMaxPool3d(output_size=1)
#         self.do = nn.Dropout3d(p=self.dropout_rate)

#     def forward(self,x,x1):
#         backbone_features = self.backbone.forward(x)
#         backbone_features=self.gmp(backbone_features)
#         backbone_features=self.do(backbone_features)
#         features=self.mlp(x1)
#         features = features.reshape(features.shape + (1,1,1))
#         global_features = torch.cat((backbone_features, features), dim=1)
#         logits = self.conv_classifier_3d(global_features)
#         logits = logits.squeeze(dim=4)
#         logits = logits.squeeze(dim=3)
#         logits = logits.squeeze(dim=2)  # squeeze will change the shape to  [batch_size, channels']

#         #cls_preds = F.softmax(logits, dim=1)
        
        
#         return logits
    
# model=FuseModelDefault(0.5,2,0.5)
#inp=torch.rand(1,2,16,128,128)
#inp2=torch.rand(1,10)
#out=model(inp,inp2)
#print(out.shape)
#out1=model(inp,inp2)
#print(out1.shape)
import monai
# model = monai.networks.nets.DenseNet121(spatial_dims=3, 
#                                         in_channels=1, 
#                                         out_channels=2)

model = monai.networks.nets.resnet10(spatial_dims=3, n_input_channels=1, n_classes=2)

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
        imge,label=data
        imge=imge.float()
        #label=label.float()
        label.to(torch.int64)
        # feature=feature.float()
        # feature=feature.to(device)
        imge=imge.to(device)
        label=label.to(device)
        #imge=imge.cuda()
        #label=label.cuda()
        # zero_out the gradient
        optimizer.zero_grad()
        output=model(imge)
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
            imge,label=data
            imge=imge.float()
            #label=label.float()
            label.to(torch.int64)
            # feature=feature.float()
            # feature=feature.to(device)
            imge=imge.to(device)
            label=label.to(device)
            #imge=imge.cuda()
            #label=label.cuda()
            output=model(imge)
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
name='3dcnnmoani'
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
