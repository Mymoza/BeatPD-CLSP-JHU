from keras.models import Model
from keras.layers import Dense, Dropout, Input, Lambda, LSTM, Masking
from keras.layers import Add

def make_DNN_model(feat_size=1200,latent_dim=30,num_layers=3,num_neuron=512,dr=0.0,hidu='relu',batch_size=50,epochs=200):

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
    encoder = Model(inputs,feats)

    return model,encoder

def make_LSTM_model(feat_size=30,num_neuron=10,out_feat_size=1,mode='regression'):
    inp = Input(shape=(None,feat_size))
    clf = Masking(mask_value=0.0)(inp)
    clf = LSTM(num_neuron)(clf)
    #clf = Dense(5,activation='softmax')(clf)    
    if mode == 'classification':
        clf = Dense(out_feat_size,activation='softmax')(clf)
    else:
        clf = Dense(out_feat_size,activation='relu')(clf)
    classifier = Model(inp,clf)
    return classifier  