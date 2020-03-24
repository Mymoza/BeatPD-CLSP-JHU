from __future__ import absolute_import
from __future__ import print_function

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda, LSTM, Masking
from keras.optimizers import RMSprop,SGD
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import to_categorical

import numpy as np
import scipy.io as sio
import h5py
import random
import glob
import os
import argparse
import pandas as pd

import tensorflow as tf
sess = tf.Session()

parser = argparse.ArgumentParser(description="Initial Experiment with Wavenet")
parser.add_argument("--data_type",default="cis")
parser.add_argument("--data_real_subtype",default="")
parser.add_argument("--subtask",default="tremor",choices=['on_off','dyskinesia', 'tremor'])
parser.add_argument("--use_ancillarydata",action="store_true")

args = parser.parse_args()

data_type = args.data_type
data_real_subtype = args.data_real_subtype
subtask = args.subtask
use_ancillarydata = args.use_ancillarydata

savedir = "/export/b03/sbhati/PD/BeatPD/Weights/"
savedir = savedir + "/" + data_type + "/" + data_real_subtype + "/_all/"

data_dir = "/home/sjoshi/codes/python/BeatPD/data/BeatPD/"
label_path_train=data_dir+data_type+'-pd.data_labels/'+data_type.upper()+'-PD_Training_Data_IDs_Labels.csv'

train_data_path = data_dir + data_type + "-pd.training_data/" + data_real_subtype + "/"
df_train_label = pd.read_csv(label_path_train)
train_data_len = df_train_label.shape[0]

if use_ancillarydata:
    ancillary_data_path = data_dir + data_type + 'pd.ancillary_data'
    label_path_ancillary=data_dir+data_type+'-pd.data_labels/'+data_type.upper()+'-PD_Ancillary_Data_IDs_Labels.csv'
    df_ancillary_label = pd.read_csv(label_path_ancillary)
    df_train_label = pd.concat([df_train_label,df_ancillary_label],axis=0,ignore_index=True)

def load_data_generator(data_frame_in):
    while True:
        #df_train_label = pd.read_csv(kfold_path+file_name)
        #N = len(df_train_label)
        for idx in data_frame_in.index:
            temp_data_path = train_data_path
            if idx >= train_data_len:
                temp_data_path = ancillary_data_apth
            #print(df_train_label["measurement_id"][idx])
            temp_train_X = pd.read_csv(temp_data_path+data_frame_in["measurement_id"][idx] + '.csv')
            temp_train_X = temp_train_X.values[:,1:]
            temp_train_X = np.log1p(temp_train_X)
            temp_train_X = temp_train_X - temp_train_X.mean(axis=0,keepdims=True)
            #import pdb; pdb.set_trace()
            temp_train_X = np.expand_dims(temp_train_X,axis=0)
            temp_train_Y = data_frame_in[subtask][idx]
            if np.isnan(temp_train_Y):
                #print('nan label')
                continue
            temp_train_Y = to_categorical(temp_train_Y,5)
            temp_train_Y = np.expand_dims(temp_train_Y,axis=0)
            yield (temp_train_X,temp_train_Y)

train_split = 0.8
N = df_train_label.shape[0]
train_ind = int(N*train_split)
ind = np.random.permutation(N)
#import pdb; pdb.set_trace()

train_df = df_train_label.iloc[ind[:train_ind]]
val_df = df_train_label.iloc[ind[train_ind:]]

#temp = df_train_label[-200:-190]
train_gen = load_data_generator(train_df)
val_gen =  load_data_generator(val_df)
train_steps = train_df.shape[0]
val_steps = val_df.shape[0]

for i in range(10):
    a,b = next(train_gen)
    #print(i)
    print(b)


from WaveNetClassifier import WaveNetClassifier

wnc = WaveNetClassifier((None,3), 5, kernel_size = 2, dilation_depth = 11, n_filters = 20, task = 'classification')

#from WaveNetClassifier_bkup import WaveNetClassifier
#wnc = WaveNetClassifier((10000,3), (5,), kernel_size = 2, dilation_depth = 11, n_filters = 40, task = 'classification')

model = wnc.get_model()

checkpointer = ModelCheckpoint(filepath=savedir+'mlp'+str(use_ancillarydata)+'.h5', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
#model.compile(optimizer='adam',loss='mse',metrics=['mse'])
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit_generator(train_gen,steps_per_epoch=train_steps,validation_data=val_gen,validation_steps=val_steps,epochs=10)
import pdb; pdb.set_trace()


