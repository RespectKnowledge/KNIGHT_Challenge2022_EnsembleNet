# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 20:46:27 2022

@author: Administrateur
"""

#%%

# Ensembling different trained models
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class EnsembledModel():

    def __init__(self, model_paths):
        super().__init__()
        self.num_models = len(model_paths)

        self.leafmodel1 = get_model_DesnNet201(model_paths[0])
        #self.leafmodel2 = get_model_Effib3(model_paths[1])
        #self.leafmodel3 = get_model_DesnNet201nn(model_paths[2])
        #self.leafmodel4 = get_model_DesnNet201n(model_paths[3])
        #self.leafmodel5 = get_model_transformer(model_paths[4])
        

    def predict(self, x):
        with torch.no_grad():
            l1 = self.leafmodel1(x)
            #l2 = self.leafmodel2(x)
            #l3=self.leafmodel3(x)
            #l4=self.leafmodel4(x)
            #l5=self.leafmodel5(x)
            #b4_e1 = self.effb4_model1(x)
            pred = (l1) / (self.num_models)
            #pred = (l1+l2+l4) 
            #pred = l4

            return pred
#%%
import os
import json
import pandas as pd
import numpy as np

#CLINICAL_NAMES = ['SubjectId', 'age', 'bmi', 'gender', 'gender_num', 'comorbidities', 'smoking_history', 'radiographic_size', 'preop_egfr',
                  #'pathology_t_stage', 'pathology_n_stage', 'pathology_m_stage','age_when_quit_smoking','pack_years', 'grade', 'aua_risk_group', 'task_1_label', 'task_2_label']

#"age_at_nephrectomy"
#"gender"
#"body_mass_index"
#"comorbidities"
#"smoking_history"
#"age_when_quit_smoking"
#"pack_years"
#"chewing_tobacco_use"
#"alcohol_use"
#"last_preop_egfr"
#"radiographic_size"
#"voxel_spacing"

CLINICAL_NAMES = ['SubjectId', 
                  'age', 
                  'bmi', 
                  'gender', 
                  'gender_num', 
                  'comorbidities', 
                  'smoking_history', 
                  'radiographic_size', 
                  'preop_egfr',
                  #'pathology_t_stage', 
                  #'pathology_n_stage', 
                  #'pathology_m_stage',
                  #'age_when_quit_smoking',
                  'alcohol_use',
                  'chewing_tobacco_use',
                  #'pack_years',
                  'x_spacing','y_spacing','z_spacing',
                  'aua_risk_group', 
                  'task_1_label', 
                  'task_2_label' ]
                  #'grade', 
                  #'aua_risk_group', 'task_1_label', 'task_2_label']

# "age_at_nephrectomy"
# "gender"
# "body_mass_index"
# "comorbidities"
# "smoking_history"
# "age_when_quit_smoking"
# "pack_years"
# "chewing_tobacco_use"
# "alcohol_use"
# "last_preop_egfr"
# "radiographic_size"
# "voxel_spacing"


def create_knight_clinical(original_file, processed_file=None):
    with open(original_file) as f:
        clinical_data = json.load(f)
        print(clinical_data)
    t_stage_count = np.zeros((5))
    aua_risk_count = np.zeros((5))
    df = pd.DataFrame(columns=CLINICAL_NAMES)
    for index, patient in enumerate(clinical_data):
        df.loc[index, 'SubjectId'] = patient['case_id']
        df.loc[index, 'age'] = patient['age_at_nephrectomy']
        df.loc[index, 'bmi'] = patient['body_mass_index']

        df.loc[index, 'gender'] = patient['gender']
        if patient['gender'] == 'male':    # 0:'male'  1:'female','transgender_male_to_female'
            df.loc[index, 'gender_num'] = 0
        else:
            df.loc[index, 'gender_num'] = 1

        df.loc[index, 'comorbidities'] = 0    # 0:no_comorbidities 1:comorbidities_exist
        for key, value in patient['comorbidities'].items():
            if value:
                df.loc[index, 'comorbidities'] = 1

        df.loc[index, 'smoking_history'] = patient['smoking_history']
        if patient['smoking_history']=='never_smoked':    # 0:'never_smoked' 1:'previous_smoker'  2:'current_smoker'
            df.loc[index, 'smoking_history'] = 0
        elif patient['smoking_history'] == 'previous_smoker':
            df.loc[index, 'smoking_history'] = 1
        elif patient['smoking_history'] == 'current_smoker':
            df.loc[index, 'smoking_history'] = 2

        df.loc[index, 'radiographic_size'] = patient['radiographic_size']
        
        # if patient['last_preop_egfr']['value'] == '>=90':
        #     df.loc[index, 'preop_egfr'] = 90
        
        # else:
        #     df.loc[index, 'preop_egfr'] = patient['last_preop_egfr']['value']
        
        #df.loc[index, 'preop_egfr'] = patient['last_preop_egfr']['value']

        #df.loc[index, 'pathology_t_stage'] = patient['pathology_t_stage']
        #df.loc[index, 'pathology_n_stage'] = patient['pathology_n_stage']
        #df.loc[index, 'pathology_m_stage'] = patient['pathology_m_stage']
        # df.loc[index, 'aua_risk_group'] = patient['aua_risk_group']
       
        # Task 1 labels:
        if patient['aua_risk_group'] in ['high_risk', 'very_high_risk']:    # 1:'3','4'  0:'0','1a','1b','2a','2b'
            df.loc[index, 'task_1_label'] = 1 # CanAT
        else:
            df.loc[index, 'task_1_label'] = 0 # NoAT

        # Task 2 labels:
        if patient['aua_risk_group']=='benign':
            df.loc[index, 'task_2_label'] = 0 
        elif patient['aua_risk_group']=='low_risk':
            df.loc[index, 'task_2_label'] = 1
        elif  patient['aua_risk_group']=='intermediate_risk':
            df.loc[index, 'task_2_label'] = 2
        elif patient['aua_risk_group']=='high_risk':
            df.loc[index, 'task_2_label'] = 3
        elif patient['aua_risk_group']=='very_high_risk':
            df.loc[index, 'task_2_label'] = 4
        else:
            ValueError('Wrong risk class')

        # former classification - deprecated
        #if patient['pathology_t_stage'] in ['3', '4']:    # 1:'3','4'  0:'0','1a','1b','2a','2b'
        #    df.loc[index, 'pathology_t_stage_classify'] = 1
        #else:
        #    df.loc[index, 'pathology_t_stage_classify'] = 0
        #t_stage = int(patient['pathology_t_stage'][0])
        #t_stage_count[t_stage] += 1
        aua_risk = int(df.loc[index, 'task_2_label'])
        aua_risk_count[aua_risk] += 1
        df.loc[index, 'grade'] = patient['tumor_isup_grade']
        ##### new parameters
        #df.loc[index,'alcohol_use']=patient['alcohol_use']
        if patient['alcohol_use']=='two_or_less_daily':
            df.loc[index, 'alcohol_use'] = 1
        elif patient['alcohol_use']=='never_or_not_in_last_3mo':
            df.loc[index, 'alcohol_use'] = 2
        elif patient['alcohol_use']=='more_than_two_daily':
            df.loc[index, 'alcohol_use'] = 3
         
        #df.loc[index,'pack_years']=patient['pack_years']
        #df.loc[index,'age_when_quit_smoking']=patient['age_when_quit_smoking']
        if patient['chewing_tobacco_use']=='never_or_not_in_last_3mo':
            df.loc[index, 'chewing_tobacco_use'] = 1
        else:
            df.loc[index, 'chewing_tobacco_use'] = 0

        #df.loc[index,'chewing_tobacco_use']=patient['chewing_tobacco_use']
        #age_when_quit_smoking
        
        if patient["last_preop_egfr"] is None:
            df.loc[index, 'preop_egfr'] = 0
        #print('yes')
        elif patient['last_preop_egfr']['value']== '>90':
            df.loc[index, 'preop_egfr'] = 90
        #df.loc[index, 'preop_egfr'] = 0
        else :
            df.loc[index, 'preop_egfr'] = patient['last_preop_egfr']['value']
        df.loc[index, 'x_spacing'] = patient['voxel_spacing']['x_spacing']
        df.loc[index, 'y_spacing'] = patient['voxel_spacing']['y_spacing']
        df.loc[index, 'z_spacing'] = patient['voxel_spacing']['z_spacing']
    # else:

    if processed_file is not None:
        # save csv file
        df.to_csv(processed_file, index=False)
        df = df.drop(['gender', 'pathology_t_stage', 'pathology_n_stage', 'pathology_m_stage'], axis=1)
        df.to_csv(os.path.splitext(processed_file)[0] + '_numeric.csv' , index=False)
    print(f'Pathology t-stage count summary: {t_stage_count}')
    print(f'AUA risk count summary: {aua_risk_count}')
    return df


original_file='C:\\Users\\Administrateur\\Desktop\\testfused_model\\features.json'

df=create_knight_clinical(original_file, processed_file=None)

dd=df['chewing_tobacco_use'].unique()

print(df['chewing_tobacco_use'].unique())


#%%
import os
import json
import pandas as pd
import numpy as np

#CLINICAL_NAMES = ['SubjectId', 'age', 'bmi', 'gender', 'gender_num', 'comorbidities', 'smoking_history', 'radiographic_size', 'preop_egfr',
                  #'pathology_t_stage', 'pathology_n_stage', 'pathology_m_stage','age_when_quit_smoking','pack_years', 'grade', 'aua_risk_group', 'task_1_label', 'task_2_label']

#"age_at_nephrectomy"
#"gender"
#"body_mass_index"
#"comorbidities"
#"smoking_history"
#"age_when_quit_smoking"
#"pack_years"
#"chewing_tobacco_use"
#"alcohol_use"
#"last_preop_egfr"
#"radiographic_size"
#"voxel_spacing"

CLINICAL_NAMES = ['SubjectId', 
                  'age', 
                  'bmi', 
                  'gender', 
                  'gender_num', 
                  'comorbidities', 
                  'smoking_history', 
                  'radiographic_size', 
                  'preop_egfr',
                  #'pathology_t_stage', 
                  #'pathology_n_stage', 
                  #'pathology_m_stage',
                  #'age_when_quit_smoking',
                  'alcohol_use',
                  'chewing_tobacco_use',
                  #'pack_years',
                  'x_spacing','y_spacing','z_spacing',
                  'aua_risk_group', 
                  'task_1_label', 
                  'task_2_label' ]
                  #'grade', 
                  #'aua_risk_group', 'task_1_label', 'task_2_label']

# "age_at_nephrectomy"
# "gender"
# "body_mass_index"
# "comorbidities"
# "smoking_history"
# "age_when_quit_smoking"
# "pack_years"
# "chewing_tobacco_use"
# "alcohol_use"
# "last_preop_egfr"
# "radiographic_size"
# "voxel_spacing"


def create_knight_clinical(original_file, processed_file=None):
    with open(original_file) as f:
        clinical_data = json.load(f)
        print(clinical_data)
    t_stage_count = np.zeros((5))
    aua_risk_count = np.zeros((5))
    df = pd.DataFrame(columns=CLINICAL_NAMES)
    for index, patient in enumerate(clinical_data):
        df.loc[index, 'SubjectId'] = patient['case_id']
        df.loc[index, 'age'] = patient['age_at_nephrectomy']
        df.loc[index, 'bmi'] = patient['body_mass_index']

        df.loc[index, 'gender'] = patient['gender']
        if patient['gender'] == 'male':    # 0:'male'  1:'female','transgender_male_to_female'
            df.loc[index, 'gender_num'] = 0
        else:
            df.loc[index, 'gender_num'] = 1

        df.loc[index, 'comorbidities'] = 0    # 0:no_comorbidities 1:comorbidities_exist
        for key, value in patient['comorbidities'].items():
            if value:
                df.loc[index, 'comorbidities'] = 1

        df.loc[index, 'smoking_history'] = patient['smoking_history']
        if patient['smoking_history']=='never_smoked':    # 0:'never_smoked' 1:'previous_smoker'  2:'current_smoker'
            df.loc[index, 'smoking_history'] = 0
        elif patient['smoking_history'] == 'previous_smoker':
            df.loc[index, 'smoking_history'] = 1
        elif patient['smoking_history'] == 'current_smoker':
            df.loc[index, 'smoking_history'] = 2

        df.loc[index, 'radiographic_size'] = patient['radiographic_size']
        
        # if patient['last_preop_egfr']['value'] == '>=90':
        #     df.loc[index, 'preop_egfr'] = 90
        
        # else:
        #     df.loc[index, 'preop_egfr'] = patient['last_preop_egfr']['value']
        
        #df.loc[index, 'preop_egfr'] = patient['last_preop_egfr']['value']

        #df.loc[index, 'pathology_t_stage'] = patient['pathology_t_stage']
        #df.loc[index, 'pathology_n_stage'] = patient['pathology_n_stage']
        #df.loc[index, 'pathology_m_stage'] = patient['pathology_m_stage']
        # df.loc[index, 'aua_risk_group'] = patient['aua_risk_group']
       
        # # Task 1 labels:
        # if patient['aua_risk_group'] in ['high_risk', 'very_high_risk']:    # 1:'3','4'  0:'0','1a','1b','2a','2b'
        #     df.loc[index, 'task_1_label'] = 1 # CanAT
        # else:
        #     df.loc[index, 'task_1_label'] = 0 # NoAT

        # # Task 2 labels:
        # if patient['aua_risk_group']=='benign':
        #     df.loc[index, 'task_2_label'] = 0 
        # elif patient['aua_risk_group']=='low_risk':
        #     df.loc[index, 'task_2_label'] = 1
        # elif  patient['aua_risk_group']=='intermediate_risk':
        #     df.loc[index, 'task_2_label'] = 2
        # elif patient['aua_risk_group']=='high_risk':
        #     df.loc[index, 'task_2_label'] = 3
        # elif patient['aua_risk_group']=='very_high_risk':
        #     df.loc[index, 'task_2_label'] = 4
        # else:
        #     ValueError('Wrong risk class')

        # former classification - deprecated
        #if patient['pathology_t_stage'] in ['3', '4']:    # 1:'3','4'  0:'0','1a','1b','2a','2b'
        #    df.loc[index, 'pathology_t_stage_classify'] = 1
        #else:
        #    df.loc[index, 'pathology_t_stage_classify'] = 0
        #t_stage = int(patient['pathology_t_stage'][0])
        #t_stage_count[t_stage] += 1
        # aua_risk = int(df.loc[index, 'task_2_label'])
        # aua_risk_count[aua_risk] += 1
        # df.loc[index, 'grade'] = patient['tumor_isup_grade']
        ##### new parameters
        #df.loc[index,'alcohol_use']=patient['alcohol_use']
        if patient['alcohol_use']=='two_or_less_daily':
            df.loc[index, 'alcohol_use'] = 1
        elif patient['alcohol_use']=='never_or_not_in_last_3mo':
            df.loc[index, 'alcohol_use'] = 2
        elif patient['alcohol_use']=='more_than_two_daily':
            df.loc[index, 'alcohol_use'] = 3
         
        #df.loc[index,'pack_years']=patient['pack_years']
        #df.loc[index,'age_when_quit_smoking']=patient['age_when_quit_smoking']
        if patient['chewing_tobacco_use']=='never_or_not_in_last_3mo':
            df.loc[index, 'chewing_tobacco_use'] = 1
        else:
            df.loc[index, 'chewing_tobacco_use'] = 0

        #df.loc[index,'chewing_tobacco_use']=patient['chewing_tobacco_use']
        #age_when_quit_smoking
        
        if patient["last_preop_egfr"] is None:
            df.loc[index, 'preop_egfr'] = 0
        #print('yes')
        elif patient['last_preop_egfr']['value']== '>90':
            df.loc[index, 'preop_egfr'] = 90
        #df.loc[index, 'preop_egfr'] = 0
        else :
            df.loc[index, 'preop_egfr'] = patient['last_preop_egfr']['value']
        df.loc[index, 'x_spacing'] = patient['voxel_spacing']['x_spacing']
        df.loc[index, 'y_spacing'] = patient['voxel_spacing']['y_spacing']
        df.loc[index, 'z_spacing'] = patient['voxel_spacing']['z_spacing']
    # else:

    if processed_file is not None:
        # save csv file
        df.to_csv(processed_file, index=False)
        df = df.drop(['gender', 'pathology_t_stage', 'pathology_n_stage', 'pathology_m_stage'], axis=1)
        df.to_csv(os.path.splitext(processed_file)[0] + '_numeric.csv' , index=False)
    print(f'Pathology t-stage count summary: {t_stage_count}')
    print(f'AUA risk count summary: {aua_risk_count}')
    return df


original_file='C:\\Users\\Administrateur\\Desktop\\testfused_model\\features.json'

df=create_knight_clinical(original_file, processed_file=None)

dd=df['chewing_tobacco_use'].unique()

print(df['chewing_tobacco_use'].unique())

#%%

import os
import json
import pandas as pd
import numpy as np

#CLINICAL_NAMES = ['SubjectId', 'age', 'bmi', 'gender', 'gender_num', 'comorbidities', 'smoking_history', 'radiographic_size', 'preop_egfr',
                  #'pathology_t_stage', 'pathology_n_stage', 'pathology_m_stage','age_when_quit_smoking','pack_years', 'grade', 'aua_risk_group', 'task_1_label', 'task_2_label']

#"age_at_nephrectomy"
#"gender"
#"body_mass_index"
#"comorbidities"
#"smoking_history"
#"age_when_quit_smoking"
#"pack_years"
#"chewing_tobacco_use"
#"alcohol_use"
#"last_preop_egfr"
#"radiographic_size"
#"voxel_spacing"

CLINICAL_NAMES = ['SubjectId', 
                  'age', 
                  'bmi', 
                  'gender', 
                  'gender_num', 
                  'comorbidities', 
                  'smoking_history', 
                  'radiographic_size', 
                  'preop_egfr',
                  #'pathology_t_stage', 
                  #'pathology_n_stage', 
                  #'pathology_m_stage',
                  #'age_when_quit_smoking',
                  'alcohol_use',
                  'chewing_tobacco_use',
                  #'pack_years',
                  'x_spacing','y_spacing','z_spacing' ]
                  #'grade', 
                  #'aua_risk_group', 'task_1_label', 'task_2_label']

# "age_at_nephrectomy"
# "gender"
# "body_mass_index"
# "comorbidities"
# "smoking_history"
# "age_when_quit_smoking"
# "pack_years"
# "chewing_tobacco_use"
# "alcohol_use"
# "last_preop_egfr"
# "radiographic_size"
# "voxel_spacing"


def create_knight_clinical(original_file, processed_file=None):
    with open(original_file) as f:
        clinical_data = json.load(f)
        print(clinical_data)
    t_stage_count = np.zeros((5))
    aua_risk_count = np.zeros((5))
    df = pd.DataFrame(columns=CLINICAL_NAMES)
    for index, patient in enumerate(clinical_data):
        df.loc[index, 'SubjectId'] = patient['case_id']
        df.loc[index, 'age'] = patient['age_at_nephrectomy']
        df.loc[index, 'bmi'] = patient['body_mass_index']

        df.loc[index, 'gender'] = patient['gender']
        if patient['gender'] == 'male':    # 0:'male'  1:'female','transgender_male_to_female'
            df.loc[index, 'gender_num'] = 0
        else:
            df.loc[index, 'gender_num'] = 1

        df.loc[index, 'comorbidities'] = 0    # 0:no_comorbidities 1:comorbidities_exist
        for key, value in patient['comorbidities'].items():
            if value:
                df.loc[index, 'comorbidities'] = 1

        df.loc[index, 'smoking_history'] = patient['smoking_history']
        if patient['smoking_history']=='never_smoked':    # 0:'never_smoked' 1:'previous_smoker'  2:'current_smoker'
            df.loc[index, 'smoking_history'] = 0
        elif patient['smoking_history'] == 'previous_smoker':
            df.loc[index, 'smoking_history'] = 1
        elif patient['smoking_history'] == 'current_smoker':
            df.loc[index, 'smoking_history'] = 2

        df.loc[index, 'radiographic_size'] = patient['radiographic_size']
        
        # if patient['last_preop_egfr']['value'] == '>=90':
        #     df.loc[index, 'preop_egfr'] = 90
        
        # else:
        #     df.loc[index, 'preop_egfr'] = patient['last_preop_egfr']['value']
        
        #df.loc[index, 'preop_egfr'] = patient['last_preop_egfr']['value']

        #df.loc[index, 'pathology_t_stage'] = patient['pathology_t_stage']
        #df.loc[index, 'pathology_n_stage'] = patient['pathology_n_stage']
        #df.loc[index, 'pathology_m_stage'] = patient['pathology_m_stage']
        # df.loc[index, 'aua_risk_group'] = patient['aua_risk_group']
       
        # # Task 1 labels:
        # if patient['aua_risk_group'] in ['high_risk', 'very_high_risk']:    # 1:'3','4'  0:'0','1a','1b','2a','2b'
        #     df.loc[index, 'task_1_label'] = 1 # CanAT
        # else:
        #     df.loc[index, 'task_1_label'] = 0 # NoAT

        # # Task 2 labels:
        # if patient['aua_risk_group']=='benign':
        #     df.loc[index, 'task_2_label'] = 0 
        # elif patient['aua_risk_group']=='low_risk':
        #     df.loc[index, 'task_2_label'] = 1
        # elif  patient['aua_risk_group']=='intermediate_risk':
        #     df.loc[index, 'task_2_label'] = 2
        # elif patient['aua_risk_group']=='high_risk':
        #     df.loc[index, 'task_2_label'] = 3
        # elif patient['aua_risk_group']=='very_high_risk':
        #     df.loc[index, 'task_2_label'] = 4
        # else:
        #     ValueError('Wrong risk class')

        # # former classification - deprecated
        # #if patient['pathology_t_stage'] in ['3', '4']:    # 1:'3','4'  0:'0','1a','1b','2a','2b'
        # #    df.loc[index, 'pathology_t_stage_classify'] = 1
        # #else:
        # #    df.loc[index, 'pathology_t_stage_classify'] = 0
        # #t_stage = int(patient['pathology_t_stage'][0])
        # #t_stage_count[t_stage] += 1
        # aua_risk = int(df.loc[index, 'task_2_label'])
        # aua_risk_count[aua_risk] += 1
        #df.loc[index, 'grade'] = patient['tumor_isup_grade']
        ##### new parameters
        #df.loc[index,'alcohol_use']=patient['alcohol_use']
        if patient['alcohol_use']=='two_or_less_daily':
            df.loc[index, 'alcohol_use'] = 1
        elif patient['alcohol_use']=='never_or_not_in_last_3mo':
            df.loc[index, 'alcohol_use'] = 2
        elif patient['alcohol_use']=='more_than_two_daily':
            df.loc[index, 'alcohol_use'] = 3
         
        #df.loc[index,'pack_years']=patient['pack_years']
        #df.loc[index,'age_when_quit_smoking']=patient['age_when_quit_smoking']
        if patient['chewing_tobacco_use']=='never_or_not_in_last_3mo':
            df.loc[index, 'chewing_tobacco_use'] = 1
        else:
            df.loc[index, 'chewing_tobacco_use'] = 0

        #df.loc[index,'chewing_tobacco_use']=patient['chewing_tobacco_use']
        #age_when_quit_smoking
        
        if patient["last_preop_egfr"] is None:
            df.loc[index, 'preop_egfr'] = 0
        #print('yes')
        elif patient['last_preop_egfr']['value']== '>90':
            df.loc[index, 'preop_egfr'] = 90
        #df.loc[index, 'preop_egfr'] = 0
        else :
            df.loc[index, 'preop_egfr'] = patient['last_preop_egfr']['value']
        df.loc[index, 'x_spacing'] = patient['voxel_spacing']['x_spacing']
        df.loc[index, 'y_spacing'] = patient['voxel_spacing']['y_spacing']
        df.loc[index, 'z_spacing'] = patient['voxel_spacing']['z_spacing']
    # else:

    if processed_file is not None:
        # save csv file
        df.to_csv(processed_file, index=False)
        df = df.drop(['gender', 'pathology_t_stage', 'pathology_n_stage', 'pathology_m_stage'], axis=1)
        df.to_csv(os.path.splitext(processed_file)[0] + '_numeric.csv' , index=False)
    print(f'Pathology t-stage count summary: {t_stage_count}')
    print(f'AUA risk count summary: {aua_risk_count}')
    return df


original_file='C:\\Users\\Administrateur\\Desktop\\testfused_model\\features.json'

df=create_knight_clinical(original_file, processed_file=None)

dd=df['chewing_tobacco_use'].unique()

print(df['chewing_tobacco_use'].unique())
# clinical_data = json.load(original_file)
# file='C:\\Users\\Administrateur\\Desktop\\testfused_model\\jsonfil\\features.json'

# with open(original_file) as f:
#   clinical_data1 = json.load(f)
#   print(clinical_data1)
 
 
# import json

# #file_path = "C:/Projects/Tryouts/books.json"

# with open(file, 'r') as j:
#      contents = json.loads(j.read())
#      print(contents)


#%%
import os
import json
import pandas as pd
import numpy as np

original_file='C:\\Users\\Administrateur\\Desktop\\testfused_model\\features.json'

#df=create_knight_clinical(original_file, processed_file=None)

# clinical_data = json.load(original_file)
# file='C:\\Users\\Administrateur\\Desktop\\testfused_model\\jsonfil\\features.json'


CLINICAL_NAMES = ['SubjectId', 
                  'age', 
                  'bmi', 
                  'gender', 
                  'gender_num', 
                  'comorbidities', 
                  'smoking_history', 
                  'radiographic_size', 
                  'preop_egfr',
                  #'pathology_t_stage', 
                  #'pathology_n_stage', 
                  #'pathology_m_stage',
                  'age_when_quit_smoking',
                  'alcohol_use',
                  'chewing_tobacco_use',
                  'pack_years','x_spacing','y_spacing','z_spacing']

with open(original_file) as f:
    clinical_data = json.load(f)
    print(clinical_data)

df = pd.DataFrame(columns=CLINICAL_NAMES)
#df = pd.DataFrame()

for index, patient in enumerate(clinical_data):
    print(patient['case_id'])
    #print(patient['last_preop_egfr'])
    #patient['last_preop_egfr']['value']
    if patient["last_preop_egfr"] is None:
        df.loc[index, 'preop_egfr'] = 0
        #print('yes')
    elif patient['last_preop_egfr']['value']== '>90':
        df.loc[index, 'preop_egfr'] = 90
        #df.loc[index, 'preop_egfr'] = 0
    else :
        df.loc[index, 'preop_egfr'] = patient['last_preop_egfr']['value']
        
    #if patient["voxel_spacing"]=='x_spacing':
    df.loc[index, 'x_spacing'] = patient['voxel_spacing']['x_spacing']
    df.loc[index, 'y_spacing'] = patient['voxel_spacing']['y_spacing']
    df.loc[index, 'z_spacing'] = patient['voxel_spacing']['z_spacing']
    # else:
    #     patient['last_preop_egfr']['value']
    #df.loc[index, 'SubjectId'] = patient['case_id']
    # if patient['last_preop_egfr']['value'] == '>90':
    #     df.loc[index, 'last_preop_egfr'] = 90
    # if ["last_preop_egfr"]=='null':
    #     df.loc[index, 'last_preop_egfr'] = 0
         
    # else:
    #     df.loc[index, 'last_preop_egfr'] = patient['last_preop_egfr']['value']
        
        

#%%
#null
def create_knight_clinical(original_file, processed_file=None):
    with open(original_file) as f:
        clinical_data = json.load(f)
        print(clinical_data)
    t_stage_count = np.zeros((5))
    aua_risk_count = np.zeros((5))
    df = pd.DataFrame(columns=CLINICAL_NAMES)
    for index, patient in enumerate(clinical_data):
        df.loc[index, 'SubjectId'] = patient['case_id']
        df.loc[index, 'age'] = patient['age_at_nephrectomy']
        df.loc[index, 'bmi'] = patient['body_mass_index']

        df.loc[index, 'gender'] = patient['gender']
        if patient['gender'] == 'male':    # 0:'male'  1:'female','transgender_male_to_female'
            df.loc[index, 'gender_num'] = 0
        else:
            df.loc[index, 'gender_num'] = 1

        df.loc[index, 'comorbidities'] = 0    # 0:no_comorbidities 1:comorbidities_exist
        for key, value in patient['comorbidities'].items():
            if value:
                df.loc[index, 'comorbidities'] = 1

        df.loc[index, 'smoking_history'] = patient['smoking_history']
        if patient['smoking_history']=='never_smoked':    # 0:'never_smoked' 1:'previous_smoker'  2:'current_smoker'
            df.loc[index, 'smoking_history'] = 0
        elif patient['smoking_history'] == 'previous_smoker':
            df.loc[index, 'smoking_history'] = 1
        elif patient['smoking_history'] == 'current_smoker':
            df.loc[index, 'smoking_history'] = 2

        df.loc[index, 'radiographic_size'] = patient['radiographic_size']
        
        # if patient['last_preop_egfr']['value'] == '>=90':
        #     df.loc[index, 'preop_egfr'] = 90
        
        # else:
        #     df.loc[index, 'preop_egfr'] = patient['last_preop_egfr']['value']
        
        #df.loc[index, 'preop_egfr'] = patient['last_preop_egfr']['value']

        #df.loc[index, 'pathology_t_stage'] = patient['pathology_t_stage']
        #df.loc[index, 'pathology_n_stage'] = patient['pathology_n_stage']
        #df.loc[index, 'pathology_m_stage'] = patient['pathology_m_stage']
        # df.loc[index, 'aua_risk_group'] = patient['aua_risk_group']
       
        # # Task 1 labels:
        # if patient['aua_risk_group'] in ['high_risk', 'very_high_risk']:    # 1:'3','4'  0:'0','1a','1b','2a','2b'
        #     df.loc[index, 'task_1_label'] = 1 # CanAT
        # else:
        #     df.loc[index, 'task_1_label'] = 0 # NoAT

        # # Task 2 labels:
        # if patient['aua_risk_group']=='benign':
        #     df.loc[index, 'task_2_label'] = 0 
        # elif patient['aua_risk_group']=='low_risk':
        #     df.loc[index, 'task_2_label'] = 1
        # elif  patient['aua_risk_group']=='intermediate_risk':
        #     df.loc[index, 'task_2_label'] = 2
        # elif patient['aua_risk_group']=='high_risk':
        #     df.loc[index, 'task_2_label'] = 3
        # elif patient['aua_risk_group']=='very_high_risk':
        #     df.loc[index, 'task_2_label'] = 4
        # else:
        #     ValueError('Wrong risk class')

        # # former classification - deprecated
        # #if patient['pathology_t_stage'] in ['3', '4']:    # 1:'3','4'  0:'0','1a','1b','2a','2b'
        # #    df.loc[index, 'pathology_t_stage_classify'] = 1
        # #else:
        # #    df.loc[index, 'pathology_t_stage_classify'] = 0
        # #t_stage = int(patient['pathology_t_stage'][0])
        # #t_stage_count[t_stage] += 1
        # aua_risk = int(df.loc[index, 'task_2_label'])
        # aua_risk_count[aua_risk] += 1
        #df.loc[index, 'grade'] = patient['tumor_isup_grade']
        ##### new parameters
        #df.loc[index,'alcohol_use']=patient['alcohol_use']
        if patient['alcohol_use']=='two_or_less_daily':
            df.loc[index, 'alcohol_use'] = 1
        elif patient['alcohol_use']=='never_or_not_in_last_3mo':
            df.loc[index, 'alcohol_use'] = 2
        elif patient['alcohol_use']=='more_than_two_daily':
            df.loc[index, 'alcohol_use'] = 3
         
        df.loc[index,'pack_years']=patient['pack_years']
        #df.loc[index,'age_when_quit_smoking']=patient['age_when_quit_smoking']
        if patient['age_when_quit_smoking']=='not_applicable':
            df.loc[index, 'age_when_quit_smoking'] = 1

        df.loc[index,'chewing_tobacco_use']=patient['chewing_tobacco_use']
        #age_when_quit_smoking


#%%
#preop_egfr
#df['preop_egfr'].unique()
df['preop_egfr']=df['preop_egfr'].fillna(77)
df['alcohol_use']=df['alcohol_use'].fillna(0)
df['pack_years']=df['pack_years'].fillna(0)
df['age_when_quit_smoking']=df['age_when_quit_smoking'].fillna(0)
df["bmi"] = df['bmi'] /50.0
df['age']=df['age']/120.0
df['preop_egfr']=df['preop_egfr']/90.0
df['radiographic_size']=df['radiographic_size']/15.0
#df['alcohol_use']=df['alcohol_use']/df['alcohol_use'].max()
df['pack_years']=df['pack_years'].astype(float) / df['pack_years'].max()

print(df['preop_egfr'].unique())
print(df['age_when_quit_smoking'].unique())
print(df['alcohol_use'].unique())
print(df['pack_years'].unique())
print(df['radiographic_size'].unique())
print(df['smoking_history'].unique())
print(df['bmi'].unique())
print(df['age'].unique())

# p="voxel_spacing": {
#       "x_spacing": 0.810546875,
#       "y_spacing": 0.810546875,
#       "z_spacing": 5.0
#     },