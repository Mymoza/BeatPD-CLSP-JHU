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

from dataload import load_data, load_data_all
from model import make_DNN_model, make_LSTM_model
from BeatPDutils import get_class_weights, sort_dict

import tensorflow as tf
sess = tf.Session()

parser = argparse.ArgumentParser(description="Initial Experiment with Wavenet")
parser.add_argument("--data_type",default="real")
parser.add_argument("--data_real_subtype",default="")
parser.add_argument("--subtask",default="tremor",choices=['on_off','dyskinesia', 'tremor'])
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

df_train_label = pd.read_csv(label_path_train)
train_data_len = df_train_label.shape[0]

def get_data_path(data_type,data_real_subtype,mode='train'):
	if mode =="train":
		temp_data_path = data_dir + data_type + "-pd.training_data/" + data_real_subtype + "/"
	else:
		temp_data_path = data_dir + data_type + "-pd.testing_data/" + data_real_subtype + "/"
	return temp_data_path

#data_real_subtypes={'smartphone_accelerometer', 'smartwatch_accelerometer', 'smartwatch_gyroscope'}
data_real_subtypes = frozenset(['smartwatch_accelerometer', 'smartwatch_gyroscope'])

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
#params['data_path'] = train_data_path
params = sort_dict(params)

cleanParams = copy.copy(params)
cleanParams['add_rotation'] = 'False'
cleanParams['add_noise'] = 'False'

smartwatch_acc_params = copy.copy(params)
smartwatch_gyro_params = copy.copy(params)

temp_path = get_data_path(data_type,'smartwatch_accelerometer','train')
smartwatch_acc_params['my_data_path'] = params.get('sw_acc_data_path',temp_path)
smartwatch_acc_params['my_mask_path'] = params.get('sw_acc_mask_path',None)

temp_path = get_data_path(data_type,'smartwatch_gyroscope','train')
smartwatch_gyro_params['my_data_path'] = params.get('sw_gyro_data_path',temp_path)
smartwatch_gyro_params['my_mask_path'] = params.get('sw_gyro_mask_path',None)

all_params = {}
all_params['smartwatch_accelerometer'] = smartwatch_acc_params
all_params['smartwatch_gyroscope'] = smartwatch_gyro_params

def load_subtype_data(data_frame_in,idx,all_params):
    temp_X = [[]] * len(data_real_subtypes)
    temp_X_lens = [[]] * len(data_real_subtypes)
    for i, data_real_subtype in enumerate(data_real_subtypes):
        #temp_path = get_data_path(data_type,data_real_subtype,mode)
        temp_X[i] = load_data(data_frame_in,idx,all_params[data_real_subtype])
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

model,encoder = make_DNN_model(feat_size=feat_size,latent_dim=latent_dim)

checkpointer = ModelCheckpoint(filepath=savedir+'mlp_AE_'+str(use_ancillarydata)+'.h5', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
#model.compile(optimizer='adam',loss='mse',metrics=['mse'])

lr=0.001
sgd = SGD(lr=lr, decay=0, momentum=0.9, nesterov=True)
model.compile(optimizer='adam',loss='mse',metrics=['mae'])

model.fit(train_X,train_X,validation_split=0.2,batch_size=batch_size,epochs=epochs,shuffle=True, verbose=1,callbacks=[checkpointer, early_stopping])

model.load_weights(savedir+'mlp_AE_'+str(use_ancillarydata)+'.h5')

encoder = Model(inputs,feats)
#encoder.save(savedir+'mlp_encoder_'+str(use_ancillarydata)+'.h5')

save_feats_path = '/export/b03/sbhati/PD/BeatPD/real_AE_feats/'
for idx in df_train_label.index:
	print(idx)
	temp_X = load_subtype_data(df_train_label,idx,all_params)
	temp_feats = encoder.predict(temp_X)
	name = df_train_label["measurement_id"][idx]     
	sio.savemat(save_feats_path+name+'.mat',{'feat':temp_feats}) 