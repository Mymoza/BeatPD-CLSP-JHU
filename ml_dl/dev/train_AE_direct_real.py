from __future__ import absolute_import
from __future__ import print_function

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda, LSTM, Masking
from keras.layers import Add
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
parser.add_argument("--data_type",default="real")
parser.add_argument("--data_real_subtype",default="")
parser.add_argument("--subtask",default="tremor",choices=['on_off','dyskinesia', 'tremor'])
parser.add_argument("--use_ancillarydata",action="store_true")

args = parser.parse_args()

data_type = args.data_type
data_real_subtype = args.data_real_subtype
subtask = args.subtask
use_ancillarydata = args.use_ancillarydata

savedir = "/export/b03/sbhati/PD/BeatPD/Weights/"
savedir = savedir + "/" + data_type + data_real_subtype + "_all/"

if not os.path.exists(savedir):
    os.mkdir(savedir)

data_dir = "/home/sjoshi/codes/python/BeatPD/data/BeatPD/"
label_path_train=data_dir+data_type+'-pd.data_labels/'+data_type.upper()+'-PD_Training_Data_IDs_Labels.csv'

df_train_label = pd.read_csv(label_path_train)
train_data_len = df_train_label.shape[0]

def get_data_path(data_type,data_real_subtype,mode='train'):
	if mode =="train":
		temp_data_path = data_dir + data_type + "-pd.training_data/" + data_real_subtype + "/"
	else:
		temp_data_path = data_dir + data_type + "-pd.testing_data/" + data_real_subtype + "/"
	return temp_data_path

#data_real_subtypes={'smartphone_accelerometer', 'smartwatch_accelerometer', 'smartwatch_gyroscope'}
data_real_subtypes={'smartwatch_accelerometer', 'smartwatch_gyroscope'}

frame_length = 400
frame_step = 200
num_layers = 3
num_neuron = 512
dr = 0.0
hidu = 'relu'
latent_dim = 30
batch_size = 500
epochs = 200

def load_data(data_frame_in,idx,temp_data_path):
    #print(df_train_label["measurement_id"][idx])
    temp_train_X = pd.read_csv(temp_data_path+data_frame_in["measurement_id"][idx] + '.csv')
    temp_train_X = temp_train_X.values[:,-3:]
    #temp_train_X = np.log1p(temp_train_X)
    #temp_train_X = temp_train_X - temp_train_X.mean(axis=0,keepdims=True)
    #import pdb; pdb.set_trace()
    sig_len = temp_train_X.shape[0]
    if sig_len < frame_length:
        temp_pad = np.zeros((frame_length+1 - sig_len,3))
        temp_train_X = np.concatenate((temp_train_X, temp_pad),axis=0)
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

def load_subtype_data(data_frame_in,idx,mode='train'):
    temp_X = [[]] * len(data_real_subtypes)
    temp_X_lens = [[]] * len(data_real_subtypes)
    for i, data_real_subtype in enumerate(data_real_subtypes):
        temp_path = get_data_path(data_type,data_real_subtype,mode)
        temp_X[i] = load_data(data_frame_in,idx,temp_path)
        temp_X_lens[i] = temp_X[i].shape[0]
        #print(temp_X.shape)
    temp_X_minlen = np.min(temp_X_lens)
    for i in range(len(data_real_subtypes)):
        temp_X[i] = temp_X[i][:temp_X_minlen,:]
    temp_X = np.hstack(temp_X)
    return temp_X 	


temp_path = get_data_path(data_type,'smartwatch_accelerometer')
file_list = os.listdir(temp_path)

def select_valid_ind(data_frame_in,file_list):
	ind = []
	for idx in data_frame_in.index:
		print(idx)
		temp_name = data_frame_in['measurement_id'][idx] + '.csv'
		if temp_name in file_list:
			ind.append(idx)
	ind = np.array(ind)
	return ind

ind = select_valid_ind(df_train_label,file_list)
df_train_label = df_train_label.iloc[ind]
df_train_label = df_train_label.reset_index(drop=True)

train_X = []

for idx in df_train_label.index:
    print(idx)
    temp_X = load_subtype_data(df_train_label,idx)
    train_X.append(temp_X)

train_X = np.vstack(train_X)

N = train_X.shape[0]
ind = np.random.permutation(N)
train_X = train_X[ind,:]

feat_size = train_X.shape[1]

def DNN_resnet_single_block(in_tensor,model_des,layer_ind):
    x = Dense(num_neuron,activation=hidu, name=model_des+str(layer_ind+1))(in_tensor)
    x = Dropout(dr)(x)
    out_tensor = Add()([in_tensor,x])
    return out_tensor   

model_des = 'encoder'
layer_ind = 0

inputs = Input(shape=(feat_size,))
for i in range(num_layers):
    if(i == 0):
        x = Dense(num_neuron, activation=hidu, name=model_des+str(layer_ind))(inputs)
        x = Dropout(dr)(x)
    else:
        #x = Dense(num_neuron, activation=hidu, name='dens_'+str(i))(x)
        #x = Dropout(dr)(x)
        x = DNN_resnet_single_block(x,model_des,layer_ind)
        layer_ind = layer_ind + 1       
    print(i)

feats = Dense(latent_dim,name='featExt')(x)

layer_ind = 0
model_des = "decoder_dense"
 
for i in range(num_layers):
    if(i == 0):
        z = Dense(num_neuron, activation=hidu, name=model_des+str(layer_ind))(feats)
        z = Dropout(dr)(z)
    else:
        #z = Dense(num_neuron, activation=hidu)(z)
        #z = Dropout(dr)(z)
        z = DNN_resnet_single_block(z,model_des,layer_ind)
        layer_ind = layer_ind + 1
    print(i)

final_out = Dense(feat_size, name=model_des+str(layer_ind+1))(z)

# encoder-decoder style?
#encoder = Model(inputs,feats)
#decoder = Model(inp, z)

model = Model(inputs,final_out)


checkpointer = ModelCheckpoint(filepath=savedir+'mlp_AE_'+str(use_ancillarydata)+'.h5', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
#model.compile(optimizer='adam',loss='mse',metrics=['mse'])

lr=0.001
sgd = SGD(lr=lr, decay=0, momentum=0.9, nesterov=True)
model.compile(optimizer='adam',loss='mse',metrics=['mae'])

#model.fit(train_X,train_X,validation_split=0.2,batch_size=batch_size,epochs=epochs,shuffle=True, verbose=1,callbacks=[checkpointer, early_stopping])

model.load_weights(savedir+'mlp_AE_'+str(use_ancillarydata)+'.h5')

encoder = Model(inputs,feats)
#encoder.save(savedir+'mlp_encoder_'+str(use_ancillarydata)+'.h5')

save_feats_path = '/export/b03/sbhati/PD/BeatPD/real_AE_feats/'
for idx in df_train_label.index:
	print(idx)
	temp_X = load_subtype_data(df_train_label,idx)
	temp_feats = encoder.predict(temp_X)
	name = df_train_label["measurement_id"][idx]     
	sio.savemat(save_feats_path+name+'.mat',{'feat':temp_feats}) 


### LSTM classifier 

def get_AE_feats(encoder,data_frame_in,mode='train'):
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
        temp_X = load_subtype_data(data_frame_in,idx,mode)
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

checkpointer = ModelCheckpoint(filepath=savedir+'mlp_LSTM_'+str(use_ancillarydata)+'_'+subtask+'.h5', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
lr=0.0001
sgd = SGD(lr=lr, decay=0, momentum=0.9, nesterov=True)

classifier.compile(optimizer='adam',loss='mse',metrics=['mae'])

classifier.fit(AE_feats,labels,validation_split=0.2,batch_size=50,epochs=100,verbose=1,callbacks=[checkpointer, early_stopping])

classifier.load_weights(savedir+'mlp_LSTM_'+str(use_ancillarydata)+'_'+subtask+'.h5')
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
tr_mse = np.sum(temp) / labels.shape[0]
tr_w_mse = np.sum(temp[:,0]*tr_class_weights[ind_selected])/np.sum(tr_class_weights[ind_selected])

#df_test_label = pd.DataFrame({'measurement_id':os.listdir(temp_path)})


label_path_test=data_dir+data_type+'-pd.data_labels/'+data_type.upper()+'-PD_Test_Data_IDs_Labels.csv'
df_test_label = pd.read_csv(label_path_test)

temp_path = get_data_path(data_type,'smartwatch_accelerometer','test')
file_list = os.listdir(temp_path)

ind = select_valid_ind(df_test_label,file_list)
df_test_label = df_test_label.iloc[ind]
df_test_label = df_test_label.reset_index(drop=True)

save_feats_path = '/export/b03/sbhati/PD/BeatPD/real_AE_feats_test/'

for idx in df_test_label.index:
	print(idx)
	temp_X = load_subtype_data(df_test_label,idx,'test')
	temp_feats = encoder.predict(temp_X)
	name = df_test_label["measurement_id"][idx]     
	sio.savemat(save_feats_path+name+'.mat',{'feat':temp_feats}) 
