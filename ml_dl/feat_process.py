import numpy as np
from dataload import load_data

def get_AE_feats(encoder,data_frame_in,subtask,params):
    AE_feats = []
    labels = []
    ind_selected = []
    lengths = []
    for idx in data_frame_in.index:
        #print(idx)   
        temp_train_Y = data_frame_in[subtask][idx]
        if np.isnan(temp_train_Y):
            print('nan label')
            continue
        temp_X = load_data(data_frame_in,idx,params)
        #temp_X = temp_X + np.random.normal(0,1,(temp_X.shape))
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
    latent_dim = temp_feats.shape[-1]
    for i in range(len(AE_feats)):
        temp_pad = np.zeros((max_len-lengths[i],latent_dim))
        AE_feats[i] = np.concatenate((AE_feats[i],temp_pad),axis=0)
        AE_feats[i] = AE_feats[i].reshape(1,-1,latent_dim)
    #
    AE_feats = np.vstack(AE_feats)
    labels = np.vstack(labels)
    ind_selected = np.array(ind_selected)
    return AE_feats, labels, ind_selected