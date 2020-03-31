import numpy as np
from scipy.spatial.transform import Rotation as R
import pandas as pd
import sys

sys.path.append('/home/sjoshi/codes/python/BeatPD/code/')
from transform_data import apply_mask

def load_data(data_frame_in,idx,params):
    #print(df_train_label["measurement_id"][idx])

    data_path = params['my_data_path']
    frame_length = params['frame_length']
    frame_step = params['frame_step']
    min_len = params['min_len']
    max_len = params['max_len']
    rot_ang = params['rot_ang']
    do_MVN = params['do_MVN']
    add_noise = params['add_noise']
    add_rotation = params['add_rotation']
    remove_inactivity = params['remove_inactivity']
    mask_path = params['my_mask_path']
    highpass_path = params['my_highpass_path']
    print(type(highpass_path))
    print('data_path before : ', data_path)
    if highpass_path is not "None": 
        print('entered the if') 
        data_path = highpass_path 
    print('data_path : ', data_path)
    #data_path="/home/sjoshi/codes/python/BeatPD/data/BeatPD/real-pd.training_data.high_pass//"
    temp_train_X = pd.read_csv(data_path+data_frame_in["measurement_id"][idx] + '.csv')
    temp_train_X = temp_train_X.values[:,-3:]
    #temp_train_X = np.log1p(temp_train_X)
    #temp_train_X = temp_train_X - temp_train_X.mean(axis=0,keepdims=True)
    #import pdb; pdb.set_trace()
    if remove_inactivity =='True':
        #mask_path=data_path[:-2]+'_mask/'
        print('mask_path : ',  mask_path)
        temp_train_X = apply_mask(data_path,
                                  data_frame_in["measurement_id"][idx],
                                  mask_path)
        temp_train_X = temp_train_X.values[:,1:]
    
    sig_len = temp_train_X.shape[0]
    if sig_len < frame_length:
        temp_pad = np.zeros((frame_length+1 - sig_len,3))
        temp_train_X = np.concatenate((temp_train_X, temp_pad),axis=0)
        sig_len = temp_train_X.shape[0]
    if add_noise == 'True':
        temp_train_X = temp_train_X + np.random.normal(0,1,(temp_train_X.shape))
    if add_rotation == 'True':
        s_ind = 0
        while (s_ind < sig_len):
            jump = np.random.randint(min_len,max_len,1)[0]
            rot = np.random.randint(-rot_ang,rot_ang,1)[0]
            r = R.from_euler('xyz', [rot]*3, degrees=True)
            rot_mat = r.as_dcm()
            temp_train_X[s_ind:s_ind+jump,:] = np.dot(temp_train_X[s_ind:s_ind+jump,:],rot_mat)
            s_ind = s_ind + jump         
    num_frames = int(np.ceil(float(np.abs(sig_len - frame_length)) / frame_step))
    pad_sig_len = num_frames * frame_step + frame_length
    temp_pad = np.zeros((pad_sig_len - sig_len,3))
    pad_sig = np.concatenate((temp_train_X, temp_pad),axis=0)
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    #temp_train_X = np.expand_dims(temp_train_X,axis=0)
    temp_train_X = temp_train_X[indices,:]
    temp_train_X = temp_train_X.reshape(temp_train_X.shape[0],-1)
    if do_MVN == 'True':
        temp_train_X = temp_train_X - temp_train_X.mean(axis=0,keepdims=True)
        temp_train_X = temp_train_X / (temp_train_X.std(axis=0,keepdims=True)+1e-9)    
    #temp_train_Y = data_frame_in[subtask][idx]
    #if np.isnan(temp_train_Y):
        #print('nan label')
    #    continue
    #temp_train_Y = to_categorical(temp_train_Y,5)
    #temp_train_Y = np.expand_dims(temp_train_Y,axis=0)
    return temp_train_X

def load_data_all(data_frame_in,params):
    train_X = []
    for idx in data_frame_in.index:
        print(idx)
        temp_X = load_data(data_frame_in,idx,params)
        train_X.append(temp_X)
    train_X = np.vstack(train_X)
    return train_X
