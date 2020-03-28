from __future__ import absolute_import
from __future__ import print_function

from keras.models import Sequential, Model, load_model
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
parser.add_argument("--subtask",default="on_off",choices=['on_off','dyskinesia', 'tremor'])
parser.add_argument("--use_ancillarydata",action="store_true")
parser.add_argument("--warmstart_LSTM",action="store_true")

args = parser.parse_args()

data_type = args.data_type
data_real_subtype = args.data_real_subtype
pid = args.pid
KFind = args.KFind
subtask = args.subtask
warmstart_LSTM = args.warmstart_LSTM

savedir = "/export/b03/sbhati/PD/BeatPD/Weights/"
load_weights_dir = savedir + "/" + data_type + data_real_subtype + "_all/"
savedir = savedir + "/" + data_type + data_real_subtype + \
        '_' + str(pid) +'_'+ str(KFind) + '_' +str(warmstart_LSTM) +"/"

if not os.path.exists(savedir):
    os.mkdir(savedir)

data_dir = "/home/sjoshi/codes/python/BeatPD/data/BeatPD/"
train_data_path = data_dir + data_type + "-pd.training_data/" + data_real_subtype + "/"

if data_type == "cis":
    kfold_path = "/home/sjoshi/codes/python/BeatPD/data/BeatPD/cis-pd.training_data.k_fold_v3/"

file_name = str(pid) + '_train_kfold_' + str(KFind) + '.csv' 
df_train_label = pd.read_csv(kfold_path+file_name)

#pids = np.array([1004,1006,1007,1019,1020,1023,1032,1034,1038,1039,1043,1044,1046,1048,1049,1051])

#for i in pids:
#    file_name = str(temp_pid) + '_train_kfold_' + str(KFind) + '.csv'

frame_length = 400
frame_step = 200
latent_dim = 30

def load_data(data_frame_in,idx,temp_data_path=train_data_path):
    #print(df_train_label["measurement_id"][idx])
    temp_train_X = pd.read_csv(temp_data_path+data_frame_in["measurement_id"][idx] + '.csv')
    temp_train_X = temp_train_X.values[:,1:]
    #temp_train_X = np.log1p(temp_train_X)
    #temp_train_X = temp_train_X - temp_train_X.mean(axis=0,keepdims=True)
    #import pdb; pdb.set_trace()
    sig_len = temp_train_X.shape[0]
    num_frames = int(np.ceil(float(np.abs(sig_len - frame_length)) / frame_step))
    pad_sig_len = num_frames * frame_step + frame_length
    temp_pad = np.zeros((pad_sig_len - sig_len,3))
    pad_sig = np.concatenate((temp_train_X, temp_pad),axis=0)
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    #temp_train_X = np.expand_dims(temp_train_X,axis=0)
    temp_train_X = temp_train_X[indices,:]
    temp_train_X = temp_train_X.reshape(temp_train_X.shape[0],-1)
    #temp_train_Y = data_frame_in[subtask][idx]
    #if np.isnan(temp_train_Y):
        #print('nan label')
    #    continue
    #temp_train_Y = to_categorical(temp_train_Y,5)
    #temp_train_Y = np.expand_dims(temp_train_Y,axis=0)
    return temp_train_X


train_X = []

for idx in df_train_label.index:
    print(idx)
    temp_X = load_data(df_train_label,idx)
    train_X.append(temp_X)

train_X = np.vstack(train_X)

N = train_X.shape[0]
ind = np.random.permutation(N)
train_X = train_X[ind,:]

feat_size = frame_length*3

#model = load_model(load_weights_dir+'mlp_AE_'+str(use_ancillarydata)+'.h5')
encoder = load_model(load_weights_dir+'mlp_encoder_False.h5')
#model.predict(train_X)

## LSTM 

def get_AE_feats(encoder,data_frame_in):
    AE_feats = []
    labels = []
    ind_selected = []
    lengths = []
    for idx in data_frame_in.index:
        print(idx)   
        temp_train_Y = data_frame_in[subtask][idx]
        if np.isnan(temp_train_Y):
            print('nan label')
            continue
        temp_X = load_data(data_frame_in,idx)
        temp_feats = encoder.predict(temp_X)
        lengths.append(temp_feats.shape[0])
        #temp_pad = np.zeros((max_len-temp_feats.shape[0],latent_dim))
        #temp_feats = np.concatenate((temp_feats,temp_pad),axis=0)
        #temp_feats = temp_feats.reshape(1,-1,latent_dim)         
        ind_selected.append(idx)    
        AE_feats.append(temp_feats)    
        #temp_train_Y = to_categorical(temp_train_Y,5)
        #temp_train_Y = np.expand_dims(temp_train_Y,axis=0)
        labels.append(temp_train_Y)
    #
    max_len = np.max(lengths)
    #
    for i in range(len(AE_feats)):
        temp_pad = np.zeros((max_len-lengths[i],latent_dim))
        AE_feats[i] = np.concatenate((AE_feats[i],temp_pad),axis=0)
        AE_feats[i] = AE_feats[i].reshape(1,-1,latent_dim)
    #
    AE_feats = np.vstack(AE_feats)
    labels = np.vstack(labels)
    ind_selected = np.array(ind_selected)
    return AE_feats, labels, ind_selected

AE_feats, labels, ind_selected = get_AE_feats(encoder,df_train_label)

inp = Input(shape=(None,latent_dim))
clf = Masking(mask_value=0.0)(inp)
clf = LSTM(10)(clf)
#clf = Dense(5,activation='softmax')(clf)
clf = Dense(1,activation='relu')(clf)
classifier = Model(inp,clf)   

#classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5)

lr=0.0001
sgd = SGD(lr=lr, decay=0, momentum=0.9, nesterov=True)

classifier.compile(optimizer='adam',loss='mse',metrics=['mae'])

if warmstart_LSTM:
    classifier.load_weights(load_weights_dir+'mlp_LSTM_False.h5')

classifier.fit(AE_feats,labels,validation_split=0.2,batch_size=20,epochs=100,shuffle=True,verbose=1,callbacks=[early_stopping])

tr_pred = classifier.predict(AE_feats,batch_size=20)

def get_class_weights(data_frame_in):
    pids = data_frame_in['subject_id'].values
    class_weights = np.zeros(np.shape(pids))
    uniq_pid = np.unique(pids)
    freq_pid = np.zeros(uniq_pid.shape)
    for i in range(len(uniq_pid)):
        freq_pid[i] = np.sqrt(np.sum(pids == uniq_pid[i]))
        class_weights[pids == uniq_pid[i]] = 1 / freq_pid[i]
    return class_weights         

tr_class_weights = get_class_weights(df_train_label)

temp = (labels-tr_pred)**2
tr_mse = np.sum(temp) / temp.shape[0]
tr_w_mse = np.sum(temp[:,0]*tr_class_weights[ind_selected])/np.sum(tr_class_weights[ind_selected])


## test 

file_name = str(pid) + '_test_kfold_' + str(KFind) + '.csv' 
df_test_label = pd.read_csv(kfold_path+file_name)


AE_feats, labels, ind_selected = get_AE_feats(encoder,df_test_label)

test_pred = classifier.predict(AE_feats,batch_size=20)

test_class_weights = get_class_weights(df_test_label)

temp = (labels-test_pred)**2
test_mse = np.sum(temp) / temp.shape[0]
test_w_mse = np.sum(temp[:,0]*test_class_weights[ind_selected])/np.sum(test_class_weights[ind_selected])

filename = savedir + 'error.txt'
f = open(filename,'w+')

f.write("Train mse: %f\r\n" % (tr_mse))
f.write("Train w_mse: %f\r\n" % (tr_w_mse))
f.write("Test mse: %f\r\n" % (test_mse))
f.write("Test w_mse: %f\r\n" % (test_w_mse))
f.write("\r\n")

f.close()