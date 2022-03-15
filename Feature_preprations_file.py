# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 14:24:30 2022

@author: Administrateur
"""
#%%

import pandas as pd
#path='C:\\Users\\Administrateur\\Desktop\\testfused_model\\fuse-med-ml-master\\fuse-med-ml-master\\baseline'
splits=pd.read_pickle('C:\\Users\\Administrateur\\Desktop\\testfused_model\\Features_models\\splits_final.pkl')
# For this example, we use split 0 out of the 5 available cross validation splits
split = splits[0]
train=split['train']
pdftrain=pd.DataFrame(train).rename(columns={0:'SubjectId'})


val=split['val']
pdfval=pd.DataFrame(val).rename(columns={0:'SubjectId'})

ptht='C:\\Users\\Administrateur\\Desktop\\testfused_model\\Features_models\\features_traindeep640.csv'
traindf=pd.read_csv(ptht)
#pathfile=traindf
traindatapd = pd.merge(traindf, pdftrain, on=['SubjectId'], how='inner')
traindatapd.head()
pthv='C:\\Users\\Administrateur\\Desktop\\testfused_model\\Features_models\\features_testdeep60.csv'
validdf=pd.read_csv(pthv)
valdatapd = pd.merge(validdf, pdfval, on=['SubjectId'], how='inner')
valdatapd.head()

###############   training features

pathclinical='C:\\Users\\Administrateur\\Desktop\\testfused_model\\Features_models\\updated_training_features.csv'
pathct=pd.read_csv(pathclinical)
list(pathct.columns)
pathct1=pathct.copy()
CLINICAL_NAMES = ['SubjectId','age','bmi',
                  'gender',
                  'gender_num',
                  'comorbidities',
                  'smoking_history',
                  'radiographic_size',
                  'preop_egfr',
                  'alcohol_use',
                  'chewing_tobacco_use',
                  'x_spacing',
                  'y_spacing',
                  'z_spacing',
                  'aua_risk_group',
                  'task_1_label',
                  'task_2_label']



#%%

import pandas as pd
#path='C:\\Users\\Administrateur\\Desktop\\testfused_model\\fuse-med-ml-master\\fuse-med-ml-master\\baseline'
splits=pd.read_pickle('C:\\Users\\Administrateur\\Desktop\\testfused_model\\Features_models\\splits_final.pkl')
# For this example, we use split 0 out of the 5 available cross validation splits
#split = splits[0]

def foldfunction(splits):
    split = splits[0]
    train=split['train']
    trainf=pd.DataFrame(train).rename(columns={0:'SubjectId'})
    val=split['val']
    testf=pd.DataFrame(val).rename(columns={0:'SubjectId'})
    
    
    return trainf,testf



def combined_fold(splits,features,labels):
    
    traindf,testdf=foldfunction(splits)
    
    trainfeatures = pd.merge(traindf, features, on=['SubjectId'], how='inner')
    validfeatures = pd.merge(testdf, features, on=['SubjectId'], how='inner')
    
    #trainclinics = pd.merge(traindf, clinicsf, on=['SubjectId'], how='inner')
    #validclinics = pd.merge(testdf, clinicsf, on=['SubjectId'], how='inner')
    
    trainlabels = pd.merge(traindf, labels, on=['SubjectId'], how='inner')
    validlabels = pd.merge(testdf, labels, on=['SubjectId'], how='inner')
    
    featuretrain = pd.merge(trainfeatures, trainlabels, on=['SubjectId'], how='inner')
    featuretest = pd.merge(validfeatures, validlabels, on=['SubjectId'], how='inner')

    return featuretrain,featuretest

# def normlaize_clinicalf(clinical):
#     clinical['preop_egfr']=clinical['preop_egfr'].fillna(77)
#     clinical["bmi"] = clinical['bmi'] /50.0
#     #df["bmi"] = df['bmi'] /50.0
#     clinical['age']=clinical['age']/120.0
#     clinical['preop_egfr']=clinical['preop_egfr']/90.0
#     clinical['radiographic_size']=clinical['radiographic_size']/15.0
#     return clinical

def correlatedFeatures(dataset, threshold):
    correlated_columns = set()
    correlations = dataset.corr()
    for i in range(len(correlations)):
        for j in range(i):
            if abs(correlations.iloc[i,j]) > threshold:
                correlated_columns.add(correlations.columns[i])
    return correlated_columns
#sel = VarianceThreshold(threshold=(0.01))
#sel.fit(X_train)

#cf = correlatedFeatures(featuretrain, 0.85)


deep=pd.read_csv('C:\\Users\\Administrateur\\Desktop\\testfused_model\\hadncrfatedfeatures\\D_WP_features_train.csv')
# def normalize(df):
#     result = df.copy()
#     for feature_name in df.columns:
#         max_value = df[feature_name].max()
#         min_value = df[feature_name].min()
#         result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
#     return result
#deep.max()
from sklearn.feature_selection import VarianceThreshold
s=deep['SubjectId']
deep = deep.drop('SubjectId', axis = 1)
print(deep.max)
m=deep.max()
deep1=deep/m.max()
#ct = correlatedFeatures(deep1, 0.85)
#deep1 = deep1.drop(ct, axis=1)
sel = VarianceThreshold(threshold=(0.4))
sel.fit(deep1)
deep1.insert(0,'SubjectId',s)

labels=pd.read_csv('C:\\Users\\Administrateur\\Desktop\\testfused_model\\Features_models\\labels_trains.csv')
# clinical=pd.read_csv('C:\\Users\\Administrateur\\Desktop\\testfused_model\\Features_models\\updated_training_features.csv')
# clinicaln=normlaize_clinicalf(clinical)
# clinical.to_csv('clincialnormalized.csv',index=False)

# def combined_fold(splits,features):
    
#     traindf,testdf=foldfunction(splits)
    
#     trainfeatures = pd.merge(traindf, features, on=['SubjectId'], how='inner')
#     validfeatures = pd.merge(testdf, features, on=['SubjectId'], how='inner')
    
#     #trainclinics = pd.merge(traindf, clinicsf, on=['SubjectId'], how='inner')
#     #validclinics = pd.merge(testdf, clinicsf, on=['SubjectId'], how='inner')
    
#     #trainlabels = pd.merge(traindf, labels, on=['SubjectId'], how='inner')
#     #validlabels = pd.merge(testdf, labels, on=['SubjectId'], how='inner')
    
#     #featuretrain = pd.merge(trainfeatures, trainlabels, on=['SubjectId'], how='inner')
#     #featuretest = pd.merge(validfeatures, validlabels, on=['SubjectId'], how='inner')

#     return trainfeatures,validfeatures

featuretrain,featuretest=combined_fold(splits,deep1,labels)

#featuretrain.to_csv('train_normalized_feat.csv',index=False)
#featuretest.to_csv('test_normalized_feat.csv',index=False)

#A_FOS_features_train

featuretrain.to_csv('D_WP_sel_train_norm.csv',index=False)
featuretest.to_csv('D_WP_sel_test_norm.csv',index=False)
#%% normalized features
import pandas as pd
train='C:\\Users\\Administrateur\\Desktop\\testfused_model\\Features_models\\best_features_sofar\\test_normalized_feat.csv'
trandf=pd.read_csv(train)
trandfs=trandf['task_2_label'].copy()

def correlatedFeatures(dataset, threshold):
    correlated_columns = set()
    correlations = dataset.corr()
    for i in range(len(correlations)):
        for j in range(i):
            if abs(correlations.iloc[i,j]) > threshold:
                correlated_columns.add(correlations.columns[i])
    return correlated_columns
trandf.drop(['task_2_label'], axis=1,inplace=True)
ct = correlatedFeatures(trandf, 0.75)
deep1 = trandf.drop(ct, axis=1)
trandf.columns

deep1.insert(15,'task_2_label',trandfs)
deep1.columns

deep1.to_csv('test_norm_sel.csv',index=False)


#%% deep features

import pandas as pd
#path='C:\\Users\\Administrateur\\Desktop\\testfused_model\\fuse-med-ml-master\\fuse-med-ml-master\\baseline'
splits=pd.read_pickle('C:\\Users\\Administrateur\\Desktop\\testfused_model\\Features_models\\splits_final.pkl')
# For this example, we use split 0 out of the 5 available cross validation splits
#split = splits[0]

def foldfunction(splits):
    split = splits[4]
    train=split['train']
    trainf=pd.DataFrame(train).rename(columns={0:'SubjectId'})
    val=split['val']
    testf=pd.DataFrame(val).rename(columns={0:'SubjectId'})
    
    
    return trainf,testf

def combined_fold(splits,features):
    
    traindf,testdf=foldfunction(splits)
    
    trainfeatures = pd.merge(traindf, features, on=['SubjectId'], how='inner')
    validfeatures = pd.merge(testdf, features, on=['SubjectId'], how='inner')
    
    #trainclinics = pd.merge(traindf, clinicsf, on=['SubjectId'], how='inner')
    #validclinics = pd.merge(testdf, clinicsf, on=['SubjectId'], how='inner')

    return trainfeatures,validfeatures
features=pd.read_csv('C:\\Users\\Administrateur\\clincialnormalized.csv')
trainfeatures,validfeatures=combined_fold(splits,features)

trainfeatures.to_csv('train_norm_fold4.csv',index=False)
validfeatures.to_csv('test_norm_fold4.csv',index=False)
#%% normalized for task2 dataset

import pandas as pd
#path='C:\\Users\\Administrateur\\Desktop\\testfused_model\\fuse-med-ml-master\\fuse-med-ml-master\\baseline'
splits=pd.read_pickle('C:\\Users\\Administrateur\\Desktop\\testfused_model\\Features_models\\splits_final.pkl')
# For this example, we use split 0 out of the 5 available cross validation splits
#split = splits[0]

def foldfunction(splits):
    split = splits[1]
    train=split['train']
    trainf=pd.DataFrame(train).rename(columns={0:'SubjectId'})
    val=split['val']
    testf=pd.DataFrame(val).rename(columns={0:'SubjectId'})
    
    
    return trainf,testf

def combined_fold(splits,features):
    
    traindf,testdf=foldfunction(splits)
    
    trainfeatures = pd.merge(traindf, features, on=['SubjectId'], how='inner')
    validfeatures = pd.merge(testdf, features, on=['SubjectId'], how='inner')
    
    #trainclinics = pd.merge(traindf, clinicsf, on=['SubjectId'], how='inner')
    #validclinics = pd.merge(testdf, clinicsf, on=['SubjectId'], how='inner')

    return trainfeatures,validfeatures
features=pd.read_csv('C:\\Users\\Administrateur\\Desktop\\testfused_model\\Features_models\\updated_training_features.csv')

import pandas as pd
def normlaize_clinicalf(clinical):
    clinical['preop_egfr']=clinical['preop_egfr'].fillna(77)
    clinical["bmi"] = clinical['bmi'] /50.0
    #df["bmi"] = df['bmi'] /50.0
    clinical['age']=clinical['age']/120.0
    clinical['preop_egfr']=clinical['preop_egfr']/90.0
    clinical['radiographic_size']=clinical['radiographic_size']/15.0
    clinical['z_spacing']=clinical['z_spacing']/5.0
    return clinical

#features_n=normlaize_clinicalf(features)


trainfeatures,validfeatures=combined_fold(splits,features)

trainfeatures.to_csv('train_un_moralize_fold1.csv',index=False)
validfeatures.to_csv('test_un_moralize_fold1.csv',index=False)





#%% test features normalized
import pandas as pd
def normlaize_clinicalf(clinical):
    clinical['preop_egfr']=clinical['preop_egfr'].fillna(77)
    clinical["bmi"] = clinical['bmi'] /50.0
    #df["bmi"] = df['bmi'] /50.0
    clinical['age']=clinical['age']/120.0
    clinical['preop_egfr']=clinical['preop_egfr']/90.0
    clinical['radiographic_size']=clinical['radiographic_size']/15.0
    return clinical

testdata=pd.read_csv('C:\\Users\\Administrateur\\Desktop\\testfused_model\\testingmodels\\testingdataset\\test_featurelatest.csv')
test_normalized=normlaize_clinicalf(testdata)
test_normalized.to_csv('test_normalized_features.csv')

#%% test for task 2

import pandas as pd
def normlaize_clinicalf(clinical):
    clinical['preop_egfr']=clinical['preop_egfr'].fillna(77)
    clinical["bmi"] = clinical['bmi'] /50.0
    #df["bmi"] = df['bmi'] /50.0
    clinical['age']=clinical['age']/120.0
    clinical['preop_egfr']=clinical['preop_egfr']/90.0
    clinical['radiographic_size']=clinical['radiographic_size']/15.0
    clinical['z_spacing']=clinical['z_spacing']/5.0
    return clinical

testdata=pd.read_csv('C:\\Users\\Administrateur\\Desktop\\testfused_model\\Features_models\\test_featuresnew.csv')
test_normalized=normlaize_clinicalf(testdata)
test_normalized.to_csv('test_normalized_wolt2.csv')



#%%

def combined_fold(splits,features,labels):
    
    traindf,testdf=foldfunction(splits)
    
    trainfeatures = pd.merge(traindf, features, on=['SubjectId'], how='inner')
    validfeatures = pd.merge(testdf, features, on=['SubjectId'], how='inner')
    
    #trainclinics = pd.merge(traindf, clinicsf, on=['SubjectId'], how='inner')
    #validclinics = pd.merge(testdf, clinicsf, on=['SubjectId'], how='inner')
    
    trainlabels = pd.merge(traindf, labels, on=['SubjectId'], how='inner')
    validlabels = pd.merge(testdf, labels, on=['SubjectId'], how='inner')
    
    featuretrain = pd.merge(trainfeatures, trainlabels, on=['SubjectId'], how='inner')
    featuretest = pd.merge(validfeatures, validlabels, on=['SubjectId'], how='inner')

    return featuretrain,featuretest


deep=pd.read_csv('C:\\Users\\Administrateur\\Desktop\\testfused_model\\Features_models\\features_testdeep60.csv')

s=deep['SubjectId']
deep = deep.drop('SubjectId', axis = 1)
print(deep.max)
m=deep.max()
deep1=deep/m.max()
deep1.insert(0,'SubjectId',s)

labels=pd.read_csv('C:\\Users\\Administrateur\\Desktop\\testfused_model\\Features_models\\labels_trains.csv')
#
featuretest = pd.merge(deep1, labels, on=['SubjectId'], how='inner')


#featuretrain,featuretest=combined_fold(splits,deep1,labels)


#featuretrain.to_csv('deepfeatures_train_norm.csv',index=False)
featuretest.to_csv('deepfeatures_test_norm.csv',index=False)
#%%
def combined_fold(splits,features):
    
    traindf,testdf=foldfunction(splits)
    
    trainfeatures = pd.merge(traindf, features, on=['SubjectId'], how='inner')
    validfeatures = pd.merge(testdf, features, on=['SubjectId'], how='inner')
    
    #trainclinics = pd.merge(traindf, clinicsf, on=['SubjectId'], how='inner')
    #validclinics = pd.merge(testdf, clinicsf, on=['SubjectId'], how='inner')

    return trainfeatures,validfeatures



#%%


def combined_fold_simple(splits,features,feature2):
    
    traindf,testdf=foldfunction(splits)
    
    trainfeatures = pd.merge(traindf, features, on=['SubjectId'], how='inner')
    validfeatures = pd.merge(testdf, features, on=['SubjectId'], how='inner')
    
    #trainclinics = pd.merge(traindf, clinicsf, on=['SubjectId'], how='inner')
    #validclinics = pd.merge(testdf, clinicsf, on=['SubjectId'], how='inner')
    
    trainlabels = pd.merge(traindf, feature2, on=['SubjectId'], how='inner')
    validlabels = pd.merge(testdf, feature2, on=['SubjectId'], how='inner')
    
    featuretrain = pd.merge(trainfeatures, trainlabels, on=['SubjectId'], how='inner')
    featuretest = pd.merge(validfeatures, validlabels, on=['SubjectId'], how='inner')

    return featuretrain,featuretest
# normalized_df=(deep-deep.min())/(deep.max()-deep.min())
# deep=normalized_df(deep)

import pandas as pd
from sklearn import preprocessing

# deep.drop(['SubjectId'],axis=1,inplace=True).values #returns a numpy array
# min_max_scaler = preprocessing.MinMaxScaler()
# x_scaled = min_max_scaler.fit_transform(deep)
# df = pd.DataFrame(x_scaled)
featuretrain,featuretest=combined_fold_simple(splits,clinical,deep1)

#list(clinical.columns)

# ['SubjectId',
#  'age',
#  'bmi',
#  'gender',
#  'gender_num',
#  'comorbidities',
#  'smoking_history',
#  'radiographic_size',
#  'preop_egfr',
#  'alcohol_use',
#  'chewing_tobacco_use',
#  'x_spacing',
#  'y_spacing',
#  'z_spacing',
#  'aua_risk_group',
#  'task_1_label',
#  'task_2_label']

#df['alcohol_use']=df['alcohol_use']/df['alcohol_use'].max()
#df['pack_years']=df['pack_years'].astype(float) / df['pack_years'].max()

#featuretrain,featuretest=combined_fold(splits,features,labels)

def combined_features_clnincs(splits,deep,clinical,labels):
    
    #clinical.drop(['SubjectId','task_1_label','task_2_label'],axis=1,inplace=True)
    f1t,f1v=combined_fold(splits,deep,labels)
    f2t,f2v=combined_fold(splits,clinical,labels)
    
    featuretrain = pd.merge(f1t, f2t, on=['SubjectId'], how='inner')
    featuretest = pd.merge(f1v, f2v, on=['SubjectId'], how='inner')
    return featuretrain,featuretest


featuretrain,featuretest=combined_features_clnincs(splits,deep,clinical,labels)

#featuretrain.drop(['SubjectId','task_1_label','task_2_label'],axis=1,inplace=True)

def combined_features_simple(splits,deep,clinical,labels):
    
    #clinical.drop(['SubjectId','task_1_label','task_2_label'],axis=1,inplace=True)
    f1t,f1v=combined_fold(splits,deep,labels)
    f2t,f2v=combined_fold(splits,clinical,labels)
    
    featuretrain = pd.merge(f1t, f2t, on=['SubjectId'], how='inner')
    featuretest = pd.merge(f1v, f2v, on=['SubjectId'], how='inner')
    return featuretrain,featuretest

#featuretrain,featuretest=combined_features_simple(splits,deep,clinical,labels)
    
    
#%%
features=pd.read_csv('C:\\Users\\Administrateur\\Desktop\\testfused_model\\hadncrfatedfeatures\\A_FOS_features_train.csv')

#trainfeatures,validfeatures=combined_features(splits,features)




CLINICAL_NAMES=['age','bmi',
                'gender',
                'gender_num',
                'comorbidities',
                'smoking_history',
                'radiographic_size',
                'preop_egfr',
                'alcohol_use',
                'chewing_tobacco_use',
                'x_spacing',
                'y_spacing',
                'z_spacing',
                'aua_risk_group']
#clinicslabels.drop(CLINICAL_NAMES,axis=1,inplace=True)

ptht='C:\\Users\\Administrateur\\Desktop\\testfused_model\\Features_models\\updated_training_features.csv'
traindf=pd.read_csv(ptht)
list(traindf.columns)
traindf.drop(CLINICAL_NAMES,axis=1,inplace=True)
traindf.to_csv('labels_trains.csv',index=False)
#pathfile=traindf
pthv='C:\\Users\\Administrateur\\Desktop\\testfused_model\\Features_models\\features_testdeep60.csv'
validdf=pd.read_csv(pthv)
df = traindf.merge(validdf,how='left', left_on='SubjectId', right_on=0)
va=[traindf,validdf]
dff=pd.concat(va)

#combined_features=pd.concate([traindf,validdf],axis=1)
result1 = pd.concat([traindf, validdf],axis=0)

cc= pd.merge(traindf, validdf, on=['SubjectId'], how='inner')
trainf=traindf
testf=validdf
trainfeatures,validfeatures=combined_features(splits,trainf,testf)

    
        
pathct1.drop(CLINICAL_NAMES,axis=1,inplace=True)
pathct1.to_csv('labelstrain.csv',index=False)

traincdeep = pd.merge(traindatapd, pathct, on=['SubjectId'], how='inner')
traincdeep.to_csv('traindeep_clinc.csv',index=False)

traindeepff = pd.merge(traindatapd, pathct1, on=['SubjectId'], how='inner')

traindeepff.to_csv('traindeep.csv',index=False)

######################## validation features
pathclinical='C:\\Users\\Administrateur\\Desktop\\testfused_model\\Features_models\\testfold\\validf0new.csv'
pathct=pd.read_csv(pathclinical)

pathct1=pathct.copy()
CLINICAL_NAMES = [
                  'age', 
                  'bmi', 
                  'gender_num', 
                  'comorbidities', 
                  'smoking_history', 
                  'radiographic_size', 
                  'preop_egfr',
                  'age_when_quit_smoking',
                  'alcohol_use',
                  'pack_years']

pathct1.drop(CLINICAL_NAMES,axis=1,inplace=True)
pathct1.to_csv('labelstrain.csv',index=False)

traincdeep = pd.merge(traindatapd, pathct, on=['SubjectId'], how='inner')
traincdeep.to_csv('traindeep_clinc.csv',index=False)

traindeepff = pd.merge(traindatapd, pathct1, on=['SubjectId'], how='inner')

traindeepff.to_csv('traindeep.csv',index=False)

