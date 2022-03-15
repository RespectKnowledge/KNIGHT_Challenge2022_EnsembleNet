# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 16:05:06 2022

@author: Administrateur
"""

from typing import Optional, Sequence
import torch.nn as nn
class ClassifierMLP(nn.Module):
    def __init__(self, in_ch: int, num_classes: Optional[int], layers_description: Sequence[int]=(256,128), dropout_rate: float = 0.1):
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
  
model1=ClassifierMLP(12,2)  
model2=ClassifierMLP(12,2) 

import torch
import torch.nn as nn
class CNN_model(nn.Module):
  def __init__(self,in_chan,classes):
    super(CNN_model,self).__init__()
    # block 1
    self.c11=nn.Conv1d(in_channels=in_chan,
                       out_channels=120,kernel_size=3) # 12-3+1=10
    #self.maxpool11=nn.MaxPool1d(2) #58
    self.c21=nn.Conv1d(in_channels=120,
                       out_channels=130,kernel_size=3) #10-3+1=8
    #self.maxpool21=nn.MaxPool1d(2) #28
    # block2
    self.c12=nn.Conv1d(in_channels=130,
                       out_channels=140,kernel_size=3) #8-3+1=6
    self.maxpool12=nn.MaxPool1d(2) #6/2=3
    self.c22=nn.Conv1d(in_channels=140,
                       out_channels=64,kernel_size=1) #3-1+1=3 
    #self.maxpool13=nn.MaxPool1d(3) #15/3=102
    # linear layer
    self.fc1=nn.Linear(64*3,128)
    self.fc2=nn.Linear(128,classes)

  def forward(self,x):
    # block 1 
    #x=self.maxpool21(self.c21(self.maxpool11(self.c11(x))))
    x=self.c12(self.c21(self.c11(x)))
    #x=(self.c22(self.maxpool12(self.c12(x))))
    x=self.maxpool12(x)
    x=self.c22(x)
    x=x.view(-1,64*3)
    x=self.fc1(x)
    x=self.fc2(x)
    return x
model3=CNN_model(in_chan=1,classes=2) # inp,number_layers,hidden_dim#
#inp=torch.rand(1,1,12)
#out=model(inp)
#print(out.shape)




import torch    

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pathmodel='C:\\Users\\Administrateur\\Desktop\\testfused_model\\testingmodels\\trained_models\\mlpmodelfold1.pth'
trainedmodel=torch.load(pathmodel,map_location=torch.device('cpu'))
model1.load_state_dict(trainedmodel)
model1.eval()
model1.to(device)
##### model 2
pathmodel2='C:\\Users\\Administrateur\\Desktop\\testfused_model\\testingmodels\\models\\fold0modelMLP\\2dlstmodel.pth'
trainedmodel2=torch.load(pathmodel2,map_location=torch.device('cpu'))
model2.load_state_dict(trainedmodel2)
model2.eval()
model2.to(device)

pathmodel3='C:\\Users\\Administrateur\\Desktop\\testfused_model\\testingmodels\\models\\1DCNN_model\\2dlstmodelCNN.pth'
trainedmodel3=torch.load(pathmodel3,map_location=torch.device('cpu'))
model3.load_state_dict(trainedmodel3)
model3.eval()
model3.to(device)

############# Ensmebling of different models ###########
def get_model_MLP1(PATH):
    model = model1
    #checkpoint = torch.load(PATH)
    #model.load_state_dict(torch.load(PATH))
    trainedmodel=torch.load(PATH,map_location=torch.device('cpu'))
    model.load_state_dict(trainedmodel)
    model.eval()
    return model.to(device)

def get_model_MLP2(PATH):
    model = model2
    #checkpoint = torch.load(PATH)
    #model.load_state_dict(torch.load(PATH))
    trainedmodel=torch.load(PATH,map_location=torch.device('cpu'))
    model.load_state_dict(trainedmodel)
    model.eval()
    return model.to(device)

def get_model_CNN1D(PATH):
    model = model3
    #checkpoint = torch.load(PATH)
    #model.load_state_dict(torch.load(PATH))
    trainedmodel=torch.load(PATH,map_location=torch.device('cpu'))
    model.load_state_dict(trainedmodel)
    model.eval()
    return model.to(device)





# Ensembling different trained models
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class EnsembledModel():

    def __init__(self, model_paths):
        super().__init__()
        self.num_models = len(model_paths)

        self.leafmodel1 = get_model_MLP2(model_paths[0])
        self.leafmodel2 = get_model_CNN1D(model_paths[1])
        self.leafmodel3 = get_model_MLP1(model_paths[2])
        #self.leafmodel4 = get_model_DesnNet201n(model_paths[3])
        #self.leafmodel5 = get_model_transformer(model_paths[4])
        

    def predict(self, x,x1,x2):
        with torch.no_grad():
            l1 = self.leafmodel1(x)
            l2 = self.leafmodel2(x1)
            l3=self.leafmodel3(x2)
            #l4=self.leafmodel4(x)
            #l5=self.leafmodel5(x)
            #b4_e1 = self.effb4_model1(x)
            pred = (l1+l2+l3) / (self.num_models)
            #pred = (l1+l2+l4) 
            #pred = l4

            return pred
        
import torch
model_paths = [
    'C:\\Users\\Administrateur\\Desktop\\testfused_model\\testingmodels\\models\\fold0modelMLP\\2dlstmodel.pth',
    'C:\\Users\\Administrateur\\Desktop\\testfused_model\\testingmodels\\models\\1DCNN_model\\2dlstmodelCNN.pth',
    'C:\\Users\\Administrateur\\Desktop\\testfused_model\\testingmodels\\trained_models\\mlpmodelfold1.pth',
    #'/content/drive/MyDrive/Models/modelDensnet220.pth',
    #'/content/drive/MyDrive/Models/model_transformers.pth',
]



model_e = EnsembledModel(model_paths)

import pandas as pd
pathdata='C:\\Users\\Administrateur\\Desktop\\testfused_model\\testingmodels\\testingdataset\\test_featurelatest.csv'
dataset2=pd.read_csv(pathdata)
#dataset2.drop("Unnamed: 0",axis=1)
pathdata1='C:\\Users\\Administrateur\\Desktop\\testfused_model\\testingmodels\\testingdataset\\test_normalized_features.csv'
dataset1=pd.read_csv(pathdata1)
dataset11=dataset1.drop("Unnamed: 0",axis=1)
#dataset1.columns


#NoAT-score
#CanAT-score
#case_id
#datatarget=datasetval['task_1_label'].copy()
import torch
#pathmodel='/content/drive/MyDrive/knight_challenegs2022/models/fold0model/3dmodeltask1.pth'
#trainedmodel=torch.load(pathmodel)
#model.load_state_dict(trainedmodel)
#model.eval()
#model.to(device)
dataframe={'case_id':[],
           'NoAT-score':[],
           'CanAT-score':[]}
CLINICAL_NAMES=['case_id','NoAT-score','CanAT-score']
CLINICAL_NAMES1=['case_id','Task1-target']
#feature=in_out.drop(['SubjectId','gender','aua_risk_group',
                          #'task_1_label','task_2_label'],axis=1, inplace=True)
import numpy as np
df = pd.DataFrame(columns=CLINICAL_NAMES)
df1=pd.DataFrame(columns=CLINICAL_NAMES1)
for i,patinet in enumerate(dataset2['SubjectId']):
  #pat=patinet.split('_')[-1]
  #pat=str(0000)+i
  number_str = str(patinet)
  pat = number_str.zfill(5)
  #print(i)
  #print(patinet)
  features_n=dataset11.iloc[i]
  #target=datasetval['task_1_label'].iloc[i]
  feature=features_n.drop(['SubjectId','gender'])
  
  datasetun=dataset2.iloc[i]
  #target=datasetval['task_1_label'].iloc[i]
  feature_un=datasetun.drop(['SubjectId','gender'])
  #feature1=feature.drop(['gender'])
  #print(feature1.shape)
  feature1=feature.copy()
  x_f_array=np.array(feature1).astype(float)
  feature1un=feature_un.copy()
  x_f_array_un=np.array(feature1un).astype(float)
  #print(x_f_array)
  #.astype(int)
  x_f_array_t=torch.from_numpy(x_f_array).float()
  x_f_array_t=torch.unsqueeze(x_f_array_t,axis=0).to(device)
  x_f_array_t2=torch.unsqueeze(x_f_array_t,axis=0).to(device)
  
  x_f_array_un_t=torch.from_numpy(x_f_array_un).float()
  x_f_array_un_t=torch.unsqueeze(x_f_array_un_t,axis=0).to(device)
  
  prediction=model_e.predict(x_f_array_t,x_f_array_t2,x_f_array_un_t)
  output=torch.softmax(prediction, dim=1)
  output=torch.squeeze(output,axis=0)
  pred=output.detach().cpu().numpy()
  #print(output.detach().cpu().numpy())
  df.loc[i,'case_id']=pat
  df.loc[i,'NoAT-score']='%.1f' % pred[0]
  df.loc[i,'CanAT-score']='%.1f' % pred[1]
  #df1.loc[i,'case_id']=pat
  #df1.loc[i,'Task1-target']=target

df.to_csv('task1_predictions_round3.csv',index=False)
#%% Task-2 prediciton


from typing import Optional, Sequence
import torch.nn as nn
class ClassifierMLP(nn.Module):
    def __init__(self, in_ch: int, num_classes: Optional[int], layers_description: Sequence[int]=(256,128), dropout_rate: float = 0.1):
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
  
model1=ClassifierMLP(12,5)  
model2=ClassifierMLP(12,5) 


import torch    

import torch
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# pathmodel='C:\\Users\\Administrateur\\Desktop\\testfused_model\\testingmodels\\trained_models\\mlpmodelfold1.pth'
# trainedmodel=torch.load(pathmodel,map_location=torch.device('cpu'))
# model1.load_state_dict(trainedmodel)
# model1.eval()
# model1.to(device)
# ##### model 2
# pathmodel2='C:\\Users\\Administrateur\\Desktop\\testfused_model\\testingmodels\\models\\fold0modelMLP\\2dlstmodel.pth'
# trainedmodel2=torch.load(pathmodel2,map_location=torch.device('cpu'))
# model2.load_state_dict(trainedmodel2)
# model2.eval()
# model2.to(device)



############# Ensmebling of different models ###########
def get_model_MLP1(PATH):
    model= model1
    #checkpoint = torch.load(PATH)
    #model.load_state_dict(torch.load(PATH))
    trainedmodel=torch.load(PATH,map_location=torch.device('cpu'))
    model.load_state_dict(trainedmodel)
    model.eval()
    return model.to(device)

def get_model_MLP2(PATH):
    model = model2
    #checkpoint = torch.load(PATH)
    #model.load_state_dict(torch.load(PATH))
    trainedmodel=torch.load(PATH,map_location=torch.device('cpu'))
    model.load_state_dict(trainedmodel)
    model.eval()
    return model.to(device)






# Ensembling different trained models
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class EnsembledModel():

    def __init__(self, model_paths):
        super().__init__()
        self.num_models = len(model_paths)

        self.leafmodel1 = get_model_MLP2(model_paths[0])
        self.leafmodel2 = get_model_MLP1(model_paths[1])
        #self.leafmodel3 = get_model_MLP1(model_paths[2])
        #self.leafmodel4 = get_model_DesnNet201n(model_paths[3])
        #self.leafmodel5 = get_model_transformer(model_paths[4])
        

    def predict(self,x):
        with torch.no_grad():
            l1 = self.leafmodel1(x)
            l2 = self.leafmodel2(x)
            #l3=self.leafmodel3(x2)
            #l4=self.leafmodel4(x)
            #l5=self.leafmodel5(x)
            #b4_e1 = self.effb4_model1(x)
            pred = (l1+l2) / (self.num_models)
            #pred = (l1+l2+l4) 
            #pred = l4

            return pred
        
import torch
model_paths = [
    'C:\\Users\\Administrateur\\Desktop\\testfused_model\\testingmodels\\task2models\\3dmodeltask2f0.pth',
    'C:\\Users\\Administrateur\\Desktop\\testfused_model\\testingmodels\\task2models\\3dmodeltask2f3.pth',
    #'C:\\Users\\Administrateur\\Desktop\\testfused_model\\testingmodels\\trained_models\\mlpmodelfold1.pth',
    #'/content/drive/MyDrive/Models/modelDensnet220.pth',
    #'/content/drive/MyDrive/Models/model_transformers.pth',
]



model_e = EnsembledModel(model_paths)

import pandas as pd
pathdata='C:\\Users\\Administrateur\\Desktop\\testfused_model\\testingmodels\\testingdataset\\test_featurelatest.csv'
dataset2=pd.read_csv(pathdata)
#dataset2.drop("Unnamed: 0",axis=1)
pathdata1='C:\\Users\\Administrateur\\Desktop\\testfused_model\\testingmodels\\testingdataset\\test_normalized_features.csv'
dataset1=pd.read_csv(pathdata1)
dataset11=dataset1.drop("Unnamed: 0",axis=1)
#dataset1.columns


#NoAT-score
#CanAT-score
#case_id
#datatarget=datasetval['task_1_label'].copy()
import torch
#pathmodel='/content/drive/MyDrive/knight_challenegs2022/models/fold0model/3dmodeltask1.pth'
#trainedmodel=torch.load(pathmodel)
#model.load_state_dict(trainedmodel)
#model.eval()
#model.to(device)
dataframe={'case_id':[],
           'NoAT-score':[],
           'CanAT-score':[]}
CLINICAL_NAMES=['case_id','B-score','LR-score','IR-score','HR-score','VHR-score']
#B-score	LR-score	IR-score	HR-score	VHR-score
#CLINICAL_NAMES1=['case_id','Task1-target']
#feature=in_out.drop(['SubjectId','gender','aua_risk_group',
                          #'task_1_label','task_2_label'],axis=1, inplace=True)
import numpy as np
df = pd.DataFrame(columns=CLINICAL_NAMES)
#df1=pd.DataFrame(columns=CLINICAL_NAMES1)
for i,patinet in enumerate(dataset2['SubjectId']):
  #pat=patinet.split('_')[-1]
  #pat=str(0000)+i
  number_str = str(patinet)
  pat = number_str.zfill(5)
  #print(i)
  #print(patinet)
  features_n=dataset11.iloc[i]
  #target=datasetval['task_1_label'].iloc[i]
  feature=features_n.drop(['SubjectId','gender'])
  
  datasetun=dataset2.iloc[i]
  #target=datasetval['task_1_label'].iloc[i]
  feature_un=datasetun.drop(['SubjectId','gender'])
  #feature1=feature.drop(['gender'])
  #print(feature1.shape)
  feature1=feature.copy()
  x_f_array=np.array(feature1).astype(float)
  feature1un=feature_un.copy()
  x_f_array_un=np.array(feature1un).astype(float)
  #print(x_f_array)
  #.astype(int)
  x_f_array_t=torch.from_numpy(x_f_array).float()
  x_f_array_t=torch.unsqueeze(x_f_array_t,axis=0).to(device)
  x_f_array_t2=torch.unsqueeze(x_f_array_t,axis=0).to(device)
  
  x_f_array_un_t=torch.from_numpy(x_f_array_un).float()
  x_f_array_un_t=torch.unsqueeze(x_f_array_un_t,axis=0).to(device)
  
  prediction=model_e.predict(x_f_array_un_t)
  output=torch.softmax(prediction, dim=1)
  output=torch.squeeze(output,axis=0)
  pred=output.detach().cpu().numpy()
  #print(output.detach().cpu().numpy())
  df.loc[i,'case_id']=pat
  df.loc[i,'B-score']='%.1f' % pred[0]
  df.loc[i,'LR-score']='%.1f' % pred[1]
  df.loc[i,'IR-score']='%.1f' % pred[2]
  df.loc[i,'HR-score']='%.1f' % pred[3]
  df.loc[i,'VHR-score']='%.1f' % pred[4]
  #df1.loc[i,'case_id']=pat
  #df1.loc[i,'Task1-target']=target

df.to_csv('task2_predictions_round3.csv',index=False)






#%%
import pandas as pd
pathdata='C:\\Users\\Administrateur\\Desktop\\testfused_model\\testingmodels\\testingdataset\\test_featuresnew.csv'
dataset2=pd.read_csv(pathdata)
#dataset2.drop("Unnamed: 0",axis=1)
pathdata1='C:\\Users\\Administrateur\\Desktop\\testfused_model\\testingmodels\\testingdataset\\test_normalized_wol.csv'
dataset1=pd.read_csv(pathdata1)
dataset11=dataset1.drop("Unnamed: 0",axis=1)
#dataset1.columns


#NoAT-score
#CanAT-score
#case_id
#datatarget=datasetval['task_1_label'].copy()
import torch
#pathmodel='/content/drive/MyDrive/knight_challenegs2022/models/fold0model/3dmodeltask1.pth'
#trainedmodel=torch.load(pathmodel)
#model.load_state_dict(trainedmodel)
#model.eval()
#model.to(device)
dataframe={'case_id':[],
           'NoAT-score':[],
           'CanAT-score':[]}
CLINICAL_NAMES=['case_id','NoAT-score','CanAT-score']
CLINICAL_NAMES1=['case_id','Task1-target']
#feature=in_out.drop(['SubjectId','gender','aua_risk_group',
                          #'task_1_label','task_2_label'],axis=1, inplace=True)
import numpy as np
df = pd.DataFrame(columns=CLINICAL_NAMES)
df1=pd.DataFrame(columns=CLINICAL_NAMES1)
for i,patinet in enumerate(dataset2['SubjectId']):
  #pat=patinet.split('_')[-1]
  #pat=str(0000)+i
  number_str = str(i)
  pat = number_str.zfill(5)
  #print(i)
  #print(patinet)
  features_n=dataset11.iloc[i]
  #target=datasetval['task_1_label'].iloc[i]
  feature=features_n.drop(['SubjectId','gender'])
  
  datasetun=dataset2.iloc[i]
  #target=datasetval['task_1_label'].iloc[i]
  feature_un=datasetun.drop(['SubjectId','gender'])
  #feature1=feature.drop(['gender'])
  #print(feature1.shape)
  feature1=feature.copy()
  x_f_array=np.array(feature1).astype(float)
  feature1un=feature_un.copy()
  x_f_array_un=np.array(feature1un).astype(float)
  #print(x_f_array)
  #.astype(int)
  x_f_array_t=torch.from_numpy(x_f_array).float()
  x_f_array_t=torch.unsqueeze(x_f_array_t,axis=0).to(device)
  
  x_f_array_un_t=torch.from_numpy(x_f_array_un).float()
  x_f_array_un_t=torch.unsqueeze(x_f_array_un_t,axis=0).to(device)
  
  prediction=model.predict(x_f_array_t,x_f_array_un_t)
  output=torch.softmax(prediction, dim=1)
  output=torch.squeeze(output,axis=0)
  pred=output.detach().cpu().numpy()
  #print(output.detach().cpu().numpy())
  df.loc[i,'case_id']=pat
  df.loc[i,'NoAT-score']='%.1f' % pred[0]
  df.loc[i,'CanAT-score']='%.1f' % pred[1]
  df1.loc[i,'case_id']=pat
  df1.loc[i,'Task1-target']=target

  #dataframe['case_id'].append(str(pat))
  #dataframe['NoAT-score'].append(pred[0])
  #dataframe['CanAT-score'].append(pred[1])
  #features.iloc[i]
  #print(x_f_array_t.shape)
  #print(pat)