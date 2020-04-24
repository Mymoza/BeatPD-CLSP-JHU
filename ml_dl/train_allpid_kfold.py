from __future__ import absolute_import
from __future__ import print_function

from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Input, Lambda, LSTM, Masking
from keras.optimizers import RMSprop,SGD
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import to_categorical

from sklearn.mixture import GaussianMixture as GMM
import numpy as np
import scipy.io as sio
import os
import argparse
import pandas as pd
import json
import copy
import pickle

from dataload import load_data
from model import make_DNN_model, make_LSTM_model, make_CNN_model
from feat_process import get_AE_feats, combine_feats
from BeatPDutils import get_class_weights, sort_dict

import tensorflow as tf
sess = tf.Session()

parser = argparse.ArgumentParser(description="Initial Experiment with Wavenet")
parser.add_argument("--data_type",default="cis")
parser.add_argument("--data_real_subtype",default="")
parser.add_argument("--pid",default=0,type=int)
parser.add_argument("--KFind",default=2,type=int)
parser.add_argument("--subtask",default="on_off",choices=['on_off','dyskinesia', 'tremor'])
parser.add_argument("-uad","--use_ancillarydata",action="store_true")
parser.add_argument("--latent_dim",default=30,type=int)
parser.add_argument("-wsLSTM","--warmstart_LSTM",action="store_true")
parser.add_argument("-dlP","--dataLoadParams",type=json.loads)
parser.add_argument("--dataAugScale",default=2,type=int)

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

savedir = "/export/b03/sbhati/PD/BeatPD/Weights/"
load_weights_AE = savedir + "/" + data_type + data_real_subtype + "_all/"
load_weights_dir = savedir + "/" + data_type + data_real_subtype + '_uad_'+ str(use_ancillarydata) +\
        '_' + subtask+'_0_'+ str(KFind) 
savedir = savedir + "/" + data_type + data_real_subtype + '_uad_'+ str(use_ancillarydata) +\
        '_' + subtask+'_'+str(pid) +'_'+ str(KFind) 

params_append_str = ""
if params:
    params = sort_dict(params)
    for key in params:
        params_append_str = params_append_str + '_' + key + '_' + str(params[key])

savedir = savedir + params_append_str + "/"
load_weights_dir = load_weights_dir + params_append_str + "/"

if not os.path.exists(savedir):
    os.mkdir(savedir)

data_dir = "/home/sjoshi/codes/python/BeatPD/data/BeatPD/"
train_data_path = data_dir + data_type + "-pd.training_data/" + data_real_subtype + "/"

if data_type == "cis":
    kfold_path = "/home/sjoshi/codes/python/BeatPD/data/BeatPD/cis-pd.training_data.k_fold_v3/"

if pid == 0:
    pids = np.array([1004,1006,1007,1019,1020,1023,1032,1034,1038,1039,1043,1044,1046,1048,1049,1051])
    df_train_label = pd.DataFrame()
    for temp_pid in pids:
        print(temp_pid)
        file_name = str(temp_pid) + '_train_kfold_' + str(KFind) + '.csv'
        temp = pd.read_csv(kfold_path+file_name)
        df_train_label = pd.concat([df_train_label,temp],axis=0,ignore_index=True)
    N = df_train_label.shape[0]
    ind = np.random.permutation(N)
    df_train_label = df_train_label.iloc[ind]
    df_train_label = df_train_label.reset_index(drop=True)
else:
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
params = sort_dict(params)

cleanParams = copy.copy(params)
cleanParams['add_rotation'] = 'False'
cleanParams['add_noise'] = 'False'

if use_ancillarydata:
    ancillary_data_path = data_dir + data_type + '-pd.ancillary_data/'
    label_path_ancillary=data_dir+data_type+'-pd.data_labels/'+data_type.upper()+'-PD_Ancillary_Data_IDs_Labels.csv'
    df_ancillary_label = pd.read_csv(label_path_ancillary)
    anci_params = copy.copy(params)
    anci_params['data_path'] = ancillary_data_path
    anci_cleanParams = copy.copy(cleanParams)
    anci_cleanParams['data_path'] = ancillary_data_path

#model = load_model(load_weights_dir+'mlp_AE_'+str(use_ancillarydata)+'.h5')
encoder = load_model(load_weights_AE+'mlp_encoder_uad_'+str(use_ancillarydata)+params_append_str+'_ld_'+str(latent_dim)+'.h5')
#encoder = load_model(load_weights_dir+'mlp_encoder_uad_'+str(use_ancillarydata)+'_ld_'+str(latent_dim)+'.h5')
#model.predict(train_X)

## LSTM 

AE_feats, labels, ind_selected = get_AE_feats(encoder,df_train_label,subtask,cleanParams)
train_data_len = AE_feats.shape[0]
if use_ancillarydata:
    acni_AE_feats, anci_labels, anci_ind_selected = get_AE_feats(encoder,df_ancillary_label,subtask,anci_cleanParams)
    AE_feats = combine_feats(AE_feats,acni_AE_feats)
    labels = np.concatenate((labels,anci_labels),axis=0)
    del acni_AE_feats,anci_labels

temp_X = AE_feats
temp_Y = labels

LSTM_featsize = AE_feats.shape[-1]
temp_X_gmm = temp_X[temp_Y[:,0]==0,:,:]
temp_X_gmm = temp_X.reshape(-1,LSTM_featsize)
ind_to_keep = np.sum(np.abs(temp_X_gmm),axis=1) > 0
temp_X_gmm = temp_X_gmm[ind_to_keep,:]
gmm = GMM(n_components=5,verbose=2,warm_start=True)
gmm.fit(temp_X_gmm)
thres = gmm.score(temp_X_gmm)
del temp_X_gmm, ind_to_keep

gmm_savename = savedir+'GMM_uad_'+str(use_ancillarydata)+'_'+subtask+params_append_str+'_ld_'+str(latent_dim)+'.sav'
pickle.dump(gmm,open(gmm_savename,'wb'))

def append_gmm_inac(gmm,in_data,thres=None):
    s1,s2,s3 = in_data.shape
    in_data = in_data.reshape(-1,s3)
    ind_to_keep = np.sum(np.abs(in_data),axis=1) > 0
    logprob = np.zeros((s1*s2,1))
    logprob[ind_to_keep,0] = gmm.score_samples(in_data[ind_to_keep,:])
    if thres is not None:
        logprob[logprob>thres] = 0
        in_data[logprob[:,0] == 0,:] = 0 
    #in_data =  np.concatenate((in_data,logprob),axis=1)
    in_data = in_data.reshape(s1,s2,s3)
    return in_data

if params['add_noise'] =='True' or params['add_rotation'] == 'True':
    for i in range(dataAugScale):
        temp_AE_feats, temp_labels, temp_ind_selected = get_AE_feats(encoder,df_train_label,subtask,params)
        if use_ancillarydata:
            acni_AE_feats, anci_labels, anci_ind_selected = get_AE_feats(encoder,df_ancillary_label,subtask,anci_cleanParams)
            temp_AE_feats = combine_feats(temp_AE_feats,acni_AE_feats)
            temp_labels = np.concatenate((temp_labels,anci_labels),axis=0)
            del acni_AE_feats,anci_labels,anci_ind_selected
        temp_X = np.concatenate((temp_AE_feats,temp_X),axis=0)
        temp_Y = np.concatenate((temp_labels,temp_Y),axis=0)
        del temp_AE_feats, temp_labels, temp_ind_selected

print("Original Size: %d" % (AE_feats.shape[0]))
print("Augumented Size: %d" % (temp_X.shape[0]))

temp_X = append_gmm_inac(gmm,temp_X,thres=thres)
ind_del = np.where(np.sum(np.abs(temp_X),axis=(1,2)) ==0)
temp_X = np.delete(temp_X,ind_del,axis=0)
temp_Y = np.delete(temp_Y,ind_del,axis=0)

#N = temp_X.shape[0]
#ind = np.random.permutation(N)
#temp_X = temp_X[ind,:,:]
#temp_Y = temp_Y[ind,:]
LSTM_featsize = temp_X.shape[-1]
classifier = make_LSTM_model(feat_size=LSTM_featsize)

checkpointer = ModelCheckpoint(filepath=savedir+'LSTM_uad_'+str(use_ancillarydata)+'_'+subtask+params_append_str+'_ld_'+str(latent_dim)+'.h5',save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

lr=0.0001
sgd = SGD(lr=lr, decay=0, momentum=0.9, nesterov=True)
classifier.compile(optimizer='adam',loss='mse',metrics=['mae'])

if warmstart_LSTM:
    lstm_loadname = load_weights_dir + 'LSTM_uad_'+str(use_ancillarydata)+'_'+subtask+params_append_str+'_ld_'+str(latent_dim)+'.h5'
    print(lstm_loadname)
    classifier.load_weights(lstm_loadname)

classifier.fit(temp_X,temp_Y,validation_split=0.10,batch_size=50,epochs=100,verbose=1,shuffle=True,callbacks=[checkpointer,early_stopping])
#classifier.fit(temp_X,temp_Y,batch_size=50,epochs=100,verbose=1,shuffle=True,callbacks=[checkpointer,early_stopping])

del temp_X, temp_Y

AE_feats_inac_appended = append_gmm_inac(gmm,AE_feats,thres=thres) 
tr_pred = classifier.predict(AE_feats_inac_appended[:train_data_len,:,:],batch_size=100,verbose=1)

tr_class_weights = get_class_weights(df_train_label.iloc[ind_selected])

temp = (labels[:train_data_len,:]-tr_pred)**2
tr_mse = np.sum(temp) / temp.shape[0]
tr_w_mse = np.sum(temp[:,0]*tr_class_weights)/np.sum(tr_class_weights)


## test 
if pid == 0:
    pids = np.array([1004,1006,1007,1019,1020,1023,1032,1034,1038,1039,1043,1044,1046,1048,1049,1051])
    df_test_label = pd.DataFrame()
    for temp_pid in pids:
        print(temp_pid)
        file_name = str(temp_pid) + '_test_kfold_' + str(KFind) + '.csv'
        temp = pd.read_csv(kfold_path+file_name)
        df_test_label = pd.concat([df_test_label,temp],axis=0,ignore_index=True)
    N = df_test_label.shape[0]
    ind = np.random.permutation(N)
    df_test_label = df_test_label.iloc[ind]
    df_test_label = df_test_label.reset_index(drop=True)
else:
    file_name = str(pid) + '_test_kfold_' + str(KFind) + '.csv' 
    df_test_label = pd.read_csv(kfold_path+file_name)


test_AE_feats, test_labels, test_ind_selected = get_AE_feats(encoder,df_test_label,subtask,params)
test_AE_feats = append_gmm_inac(gmm,test_AE_feats,thres=thres)

test_pred = classifier.predict(test_AE_feats,batch_size=100,verbose=1)

test_class_weights = get_class_weights(df_test_label.iloc[test_ind_selected])

temp = (test_labels-test_pred)**2
test_mse = np.sum(temp) / temp.shape[0]
test_w_mse = np.sum(temp[:,0]*test_class_weights)/np.sum(test_class_weights)

temp = (test_labels-labels.mean())**2
test_ml_mse = np.sum(temp) / temp.shape[0]
test_ml_w_mse = np.sum(temp[:,0]*test_class_weights)/np.sum(test_class_weights)


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

'''

save_feats_path = '/export/b03/sbhati/PD/BeatPD/AE_feats_inac_removed/'
for idx in df_train_label.index:
    print(idx)
    temp_X = load_data(df_train_label,idx,cleanParams)
    temp_feats = encoder.predict(temp_X)
    name = df_train_label["measurement_id"][idx]
    logprob = gmm.score_samples(temp_feats)
    temp_feats = temp_feats[logprob<thres,:]
    sio.savemat(save_feats_path+name+'.mat',{'feat':temp_feats})

save_feats_path = '/export/b03/sbhati/PD/BeatPD/AE_feats_inac_removed/'
for idx in df_test_label.index:
    print(idx)
    temp_X = load_data(df_test_label,idx,cleanParams)
    temp_feats = encoder.predict(temp_X)
    name = df_test_label["measurement_id"][idx]
    logprob = gmm.score_samples(temp_feats)
    temp_feats = temp_feats[logprob<thres,:]
    sio.savemat(save_feats_path+name+'.mat',{'feat':temp_feats})

'''    