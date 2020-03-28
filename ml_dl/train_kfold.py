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
parser.add_argument("--pid",default=1039,type=int)
parser.add_argument("--KFind",default=2,type=int)
parser.add_argument("--subtask",default="on_off",choices=['on_off','dyskinesia', 'tremor'])
parser.add_argument("--use_ancillarydata",action="store_true")
parser.add_argument("--latent_dim",default=30,type=int)
parser.add_argument("-wsLSTM","--warmstart_LSTM",action="store_true")
parser.add_argument("-dlP","--dataLoadParams",type=json.loads)
parser.add_argument("--dataAugScale",default=5,type=int)

args = parser.parse_args()

data_type = args.data_type
data_real_subtype = args.data_real_subtype
pid = args.pid
KFind = args.KFind
subtask = args.subtask
use_ancillarydata = args.use_ancillarydata
latent_dim = args.latent_dim
warmstart_LSTM = args.warmstart_LSTM
params = args.dataLoadParams
dataAugScale = args.dataAugScale

savedir = "/export/b19/mpgill/BeatPD/Weights/"
load_weights_dir = savedir + "/" + data_type + data_real_subtype + "_all/"
savedir = savedir + "/" + data_type + data_real_subtype + '_uad_'+ str(use_ancillarydata) +\
        '_' + str(pid) +'_'+ str(KFind) + '_wsLSTM_' +str(warmstart_LSTM) 

params_append_str = ""
if params:
    params = sort_dict(params)
    for key in params:
        params_append_str = params_append_str + '_' + key + '_' + str(params[key])

savedir = savedir + params_append_str + "/"

if not os.path.exists(savedir):
    os.mkdir(savedir)

data_dir = "/home/sjoshi/codes/python/BeatPD/data/BeatPD/"
train_data_path = data_dir + data_type + "-pd.training_data/" + data_real_subtype + "/"

if data_type == "cis":
    kfold_path = "/home/sjoshi/codes/python/BeatPD/data/BeatPD/cis-pd.training_data.k_fold_v3/"

file_name = str(pid) + '_train_kfold_' + str(KFind) + '.csv' 
df_train_label = pd.read_csv(kfold_path+file_name)

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
params['remove_inactivity'] = params.get('remove_inactivity','True')
params = sort_dict(params)

cleanParams = copy.copy(params)
cleanParams['add_rotation'] = 'False'
cleanParams['add_noise'] = 'False'

#model = load_model(load_weights_dir+'mlp_AE_'+str(use_ancillarydata)+'.h5')
encoder = load_model(load_weights_dir+'mlp_encoder_uad_'+str(use_ancillarydata)+params_append_str+'_ld_'+str(latent_dim)+'.h5')
#model.predict(train_X)

## LSTM 

AE_feats, labels, ind_selected = get_AE_feats(encoder,df_train_label,subtask,cleanParams)

LSTM_featsize = AE_feats.shape[-1]
classifier = make_LSTM_model(feat_size=LSTM_featsize)

early_stopping = EarlyStopping(monitor='val_loss', patience=5)

lr=0.0001
sgd = SGD(lr=lr, decay=0, momentum=0.9, nesterov=True)
classifier.compile(optimizer='adam',loss='mse',metrics=['mae'])

if warmstart_LSTM:
    lstm_loadname = load_weights_dir + 'LSTM_uad_'+str(use_ancillarydata)+'_'+subtask+params_append_str+'_ld_'+str(latent_dim)+'.h5'
    print(lstm_loadname)
    classifier.load_weights(lstm_loadname)

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

#import pdb; pdb.set_trace()

#N = temp_X.shape[0]
#ind = np.random.permutation(N)
#temp_X = temp_X[ind,:,:]
#temp_Y = temp_Y[ind,:]
classifier.fit(temp_X,temp_Y,validation_split=0.15,batch_size=50,epochs=100,verbose=1,shuffle=True,callbacks=[early_stopping])

del temp_X, temp_Y

tr_pred = classifier.predict(AE_feats,batch_size=20)

tr_class_weights = get_class_weights(df_train_label)

temp = (labels-tr_pred)**2
tr_mse = np.sum(temp) / temp.shape[0]
tr_w_mse = np.sum(temp[:,0]*tr_class_weights[ind_selected])/np.sum(tr_class_weights[ind_selected])


## test 

file_name = str(pid) + '_test_kfold_' + str(KFind) + '.csv' 
df_test_label = pd.read_csv(kfold_path+file_name)


test_AE_feats, test_labels, test_ind_selected = get_AE_feats(encoder,df_test_label,subtask,params)

test_pred = classifier.predict(test_AE_feats,batch_size=20)

test_class_weights = get_class_weights(df_test_label)

temp = (test_labels-test_pred)**2
test_mse = np.sum(temp) / temp.shape[0]
test_w_mse = np.sum(temp[:,0]*test_class_weights[test_ind_selected])/np.sum(test_class_weights[test_ind_selected])

temp = (test_labels-test_labels.mean())**2
test_ml_mse = np.sum(temp) / temp.shape[0]
test_ml_w_mse = np.sum(temp[:,0]*test_class_weights[test_ind_selected])/np.sum(test_class_weights[test_ind_selected])


filename = savedir + 'error.txt'
f = open(filename,'w+')

f.write("Train mse: %f\r\n" % (tr_mse))
f.write("Train w_mse: %f\r\n" % (tr_w_mse))
f.write("Test mse: %f\r\n" % (test_mse))
f.write("Test w_mse: %f\r\n" % (test_w_mse))
f.write("Test mean label mse: %f\r\n" % (test_ml_mse))
f.write("Test mean label w_mse: %f\r\n" % (test_ml_w_mse))
f.write("\r\n")

f.close()

filename = savedir + 'params.txt'
f = open(filename,'w+')

for key in params:
    f.write(key + ': ' +str(params[key]) + '\r\n')

f.close()

