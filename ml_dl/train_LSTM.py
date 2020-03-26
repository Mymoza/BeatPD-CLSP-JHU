from __future__ import absolute_import
from __future__ import print_function

from keras.models import Sequential, Model, load_model
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
parser.add_argument("-dlP","--dataLoadParams",type=json.loads)
parser.add_argument("--dataAugScale",default=2,type=int)

args = parser.parse_args()

data_type = args.data_type
data_real_subtype = args.data_real_subtype
subtask = args.subtask
use_ancillarydata = args.use_ancillarydata
latent_dim = args.latent_dim
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

#model.load_weights(savedir+'mlp_AE_uad_'+str(use_ancillarydata)+'.h5') 
#encoder.save(savedir+'mlp_encoder_uad_'+str(use_ancillarydata)+'.h5')

#encoder.load_weights(savedir+'mlp_encoder_uad_'+str(use_ancillarydata)+'.h5')

#encoder = load_model(savedir+'mlp_encoder_uad_'+str(use_ancillarydata)+params_append_str+'_ld_'+str(latent_dim)+'.h5')
encoder = load_model(savedir+'mlp_encoder_uad_'+str(use_ancillarydata)+'_ld_'+str(latent_dim)+'.h5')

### LSTM classifier 

AE_feats, labels, ind_selected = get_AE_feats(encoder,df_train_label,subtask,cleanParams)

LSTM_featsize = AE_feats.shape[-1]
classifier = make_LSTM_model(feat_size=LSTM_featsize)

#classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

lstm_savename = savedir+'LSTM_uad_'+str(use_ancillarydata)+'_'+subtask+params_append_str+'_ld_'+str(latent_dim)+'.h5'
checkpointer = ModelCheckpoint(filepath=lstm_savename, verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

lr=0.0001
sgd = SGD(lr=lr, decay=0, momentum=0.9, nesterov=True)

classifier.compile(optimizer='adam',loss='mse',metrics=['mae'])
#classifier.fit(AE_feats,labels,validation_split=0.2,batch_size=50,epochs=100,verbose=1,callbacks=[checkpointer, early_stopping])

temp_X = AE_feats
temp_Y = labels

if params['add_noise'] =='True' or params['add_rotation'] == 'True':
    for i in range(dataAugScale):
        temp_AE_feats, temp_labels, temp_ind_selected = get_AE_feats(encoder,df_train_label,subtask,params)
        temp_X = np.concatenate((temp_X,temp_AE_feats),axis=0)
        temp_Y = np.concatenate((temp_Y,temp_labels),axis=0)
        del temp_AE_feats, temp_labels, temp_ind_selected

print("Original Size: %f" % (AE_feats.shape[0]))
print("Augumented Size: %f" % (temp_X.shape[0]))

#N = temp_X.shape[0]
#ind = np.random.permutation(N)
#temp_X = temp_X[ind,:,:]
#temp_Y = temp_Y[ind,:]
classifier.fit(temp_X,temp_Y,validation_split=0.15,batch_size=50,epochs=100,verbose=1,shuffle=True,callbacks=[checkpointer, early_stopping])

tr_pred = classifier.predict(AE_feats,batch_size=20)

tr_class_weights = get_class_weights(df_train_label)

temp = (labels-tr_pred)**2
tr_mse = np.sum(temp) / temp.shape[0]
tr_w_mse = np.sum(temp[:,0]*tr_class_weights[ind_selected])/np.sum(tr_class_weights[ind_selected])

print("Train mse: %f\r\n" % (tr_mse))
print("Train w_mse: %f\r\n" % (tr_w_mse))
#### 
