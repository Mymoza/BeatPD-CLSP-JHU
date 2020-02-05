#!/usr/bin/env python
# coding: utf-8

# # BEAT-PD Challenge
# 
# Challenge website : https://www.synapse.org/#!Synapse:syn20825169/wiki/596118
# 
# Data information : https://www.synapse.org/#!Synapse:syn20825169/wiki/600405
# 

# ### Ideas/Doubts [Laureano]
# 
# VAD like thing to remove unwanted data?
# modified MFCC?
# X,Y,Z = relative positions or acceleration?
# 
# Imp: Predict per person. Maybe UBM like thing and adapt it


# Import required libraries

import pandas as pd
from IPython.display import display, HTML


# Data paths

data_dir='/home/sjoshi/codes/python/BeatPD/data/BeatPD/'


# Setup file names

'''
data_type={cis , real}

If data_type is real, data_real_subtype 
data_real_subtype={smartphone_accelerometer , smartwatch_accelerometer , smartwatch_gyroscope}
'''

#data_type='cis' 
data_type='real' 
#data_real_subtype='smartphone_accelerometer' 
#data_real_subtype='smartwatch_accelerometer'
data_real_subtype='smartwatch_gyroscope'

if data_type=='cis':
    path_train_labels=data_dir+data_type+'-pd.data_labels/'+data_type.upper()+'-PD_Training_Data_IDs_Labels.csv'
    path_train_data=data_dir+data_type+'-pd.training_data/'
    
if data_type=='real':
    path_train_labels=data_dir+data_type+'-pd.data_labels/'+data_type.upper()+'-PD_Training_Data_IDs_Labels.csv'
    path_train_data=data_dir+data_type+'-pd.training_data/'+data_real_subtype+'/'

# Display labels

df_train_label=pd.read_csv(path_train_labels)
#display(df_train_label)



#Iterating through training csv files

#for idx in range(1,len(df_train_label)):
for idx in range(1,2): #[!! Only reading first file to see how data is organized!!]
    df_train_data=pd.read_csv(path_train_data+df_train_label["measurement_id"][idx]+'.csv')
    #display(df_train_data)





