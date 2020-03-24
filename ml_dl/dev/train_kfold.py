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
parser.add_argument("--pid",default=1039,type=int)
parser.add_argument("--KFind",default=2,type=int)
parser.add_argument("--subtask",default="tremor",choices=['on_off','dyskinesia', 'tremor'])

args = parser.parse_args()

data_type = args.data_type
data_real_subtype = args.data_real_subtype
pid = args.pid
KFind = args.KFind
subtask = args.subtask

savedir = "/export/b03/sbhati/PD/BeatPD/Weights/"
savedir = savedir + "/" + data_type + data_real_subtype + \
        str(pid) + str(KFind) + "/"

data_dir = "/home/sjoshi/codes/python/BeatPD/data/BeatPD/"
data_path = data_dir + data_type + "-pd.training_data/" + data_real_subtype + "/"

if data_type == "cis":
    kfold_path = "/home/sjoshi/codes/python/BeatPD/data/BeatPD/cis-pd.training_data.k_fold_v2/"

file_name = str(pid) + '_train_kfold_' + str(KFind) + '.csv' 
df_train_label = pd.read_csv(kfold_path+file_name)

#pids = np.array([1004,1006,1007,1019,1020,1023,1032,1034,1038,1039,1043,1044,1046,1048,1049,1051])

#for i in pids:
#    file_name = str(temp_pid) + '_train_kfold_' + str(KFind) + '.csv'

def load_data_generator(df_train_label):
    while True:
        #df_train_label = pd.read_csv(kfold_path+file_name)
        N = len(df_train_label)
        for idx in df_train_label.index:
            #import pdb; pdb.set_trace()
            #idx = 1
            #print(df_train_label["measurement_id"][idx])
            temp_train_X = pd.read_csv(data_path+df_train_label["measurement_id"][idx] + '.csv')
            temp_train_X = temp_train_X.values[:,1:]
            temp_train_X = np.expand_dims(temp_train_X,axis=0)
            temp_train_Y = df_train_label[subtask][idx]
            if np.isnan(temp_train_Y):
                print('nan label')
                continue
            temp_train_Y = to_categorical(temp_train_Y,5)
            temp_train_Y = np.expand_dims(temp_train_Y,axis=0)
            yield (temp_train_X,temp_train_Y)

tg = load_data_generator(df_train_label)
for i in range(5):
    a,b = next(tg)
    #print(i)
    print(b.shape)


from WaveNetClassifier import WaveNetClassifier
wnc = WaveNetClassifier((None,3), 5, kernel_size = 2, dilation_depth = 9, n_filters = 40, task = 'classification')

model = wnc.get_model()

#model.compile(optimizer='adam',loss='mse',metrics=['mse'])
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit_generator(tg,steps_per_epoch=40,epochs=10)

#wnc.fit(a,b)


