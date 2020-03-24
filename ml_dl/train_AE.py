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
import os
import argparse
import pandas as pd
import json
import copy

from dataload import load_data
from model import make_DNN_model, make_LSTM_model
from feat_process import get_AE_feats
from BeatPDutils import get_class_weights, sort_dict

import tensorflow as tf
sess = tf.Session()

parser = argparse.ArgumentParser(description="Initial Experiment with Wavenet")
parser.add_argument("--data_type",default="cis")
parser.add_argument("--data_real_subtype",default="")
parser.add_argument("--subtask",default="on_off",choices=['on_off','dyskinesia', 'tremor'])
parser.add_argument("-uad","--use_ancillarydata",action="store_true")
parser.add_argument("--latent_dim",default=30,type=int)
parser.add_argument("--saveAEFeats",action="store_true")
parser.add_argument("-dlP","--dataLoadParams",type=json.loads)
parser.add_argument("--dataAugScale",default=2,type=int)

args = parser.parse_args()

data_type = args.data_type
data_real_subtype = args.data_real_subtype
subtask = args.subtask
use_ancillarydata = args.use_ancillarydata
latent_dim = args.latent_dim
saveAEFeats = args.saveAEFeats
params = args.dataLoadParams
dataAugScale = args.dataAugScale

savedir = "/export/b03/sbhati/PD/BeatPD/Weights/"
savedir = savedir + "/" + data_type + data_real_subtype + "_all/"

if not os.path.exists(savedir):
    os.mkdir(savedir)

params_append_str = ""
if params:
    params = sort_dict(params)
    for key in params:
        params_append_str = params_append_str + '_' + key + '_' + str(params[key])

data_dir = "/home/sjoshi/codes/python/BeatPD/data/BeatPD/"
label_path_train=data_dir+data_type+'-pd.data_labels/'+data_type.upper()+'-PD_Training_Data_IDs_Labels.csv'

train_data_path = data_dir + data_type + "-pd.training_data/" + data_real_subtype + "/"
df_train_label = pd.read_csv(label_path_train)
train_data_len = df_train_label.shape[0]

if use_ancillarydata:
    ancillary_data_path = data_dir + data_type + 'pd.ancillary_data'
    label_path_ancillary=data_dir+data_type+'-pd.data_labels/'+data_type.upper()+'-PD_Ancillary_Data_IDs_Labels.csv'
    df_ancillary_label = pd.read_csv(label_path_ancillary)
    #df_train_label = pd.concat([df_train_label,df_ancillary_label],axis=0,ignore_index=True)

if not params:
    params = {}

params['frame_length'] = params.get('frame_length',400)
params['frame_step'] = params.get('frame_step',200)
params['min_len'] = params.get('min_len',1000)
params['max_len'] = params.get('max_len',10000)
params['rot_ang'] = params.get('rot_ang',15)
params['do_MVN'] = params.get('do_MVN','False')
params['add_rotation'] = params.get('add_rotation','False')
params['add_noise'] = params.get('add_noise','False')
params['data_path'] = train_data_path
params = sort_dict(params)

cleanParams = copy.copy(params)
cleanParams['add_rotation'] = 'False'
cleanParams['add_noise'] = 'False'

train_X = []
for idx in df_train_label.index:
    print(idx)
    temp_X = load_data(df_train_label,idx,cleanParams)
    train_X.append(temp_X)

train_X = np.vstack(train_X)

N = train_X.shape[0]
ind = np.random.permutation(N)
train_X = train_X[ind,:]

model,encoder = make_DNN_model(feat_size=params['frame_length']*3,latent_dim=latent_dim)

checkpointer = ModelCheckpoint(filepath=savedir+'mlp_AE_uad_'+str(use_ancillarydata)+params_append_str+'_ld_'+str(latent_dim)+'.h5', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
#model.compile(optimizer='adam',loss='mse',metrics=['mse'])

lr=0.001
sgd = SGD(lr=lr, decay=0, momentum=0.9, nesterov=True)
model.compile(optimizer='adam',loss='mse',metrics=['mae'])

batch_size = 500
epochs = 200
model.fit(train_X,train_X,validation_split=0.2,batch_size=batch_size,epochs=epochs,shuffle=True, verbose=1,callbacks=[checkpointer, early_stopping])

model.load_weights(savedir+'mlp_AE_uad_'+str(use_ancillarydata)+params_append_str+'_ld_'+str(latent_dim)+'.h5') 
encoder.save(savedir+'mlp_encoder_uad_'+str(use_ancillarydata)+params_append_str+'_ld_'+str(latent_dim)+'.h5')

encoder.load_weights(savedir+'mlp_encoder_uad_'+str(use_ancillarydata)+params_append_str+'_ld_'+str(latent_dim)+'.h5')

if saveAEFeats:
    save_feats_path = '/export/b03/sbhati/PD/BeatPD/AE_feats/'
    for idx in df_train_label.index:
        print(idx)
        temp_X = load_data(df_train_label,idx)
        temp_feats = encoder.predict(temp_X)
        name = df_train_label["measurement_id"][idx]
        sio.savemat(save_feats_path+name+'.mat',{'feat':temp_feats}) 






