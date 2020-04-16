#!/usr/bin/env python3.7
import os
import sys
sys.path.insert(0, "/export/c10/lmorove1/PythonLibs/plda")
import plda
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import kaldi_io
import argparse
import pickle
from sklearn.decomposition import PCA
import pandas as pd
import re 

# SVR
from sklearn.svm import SVR

# KNN
from sklearn.neighbors import KNeighborsClassifier

# MSE 
from sklearn.metrics import mean_squared_error

from get_final_scores_accuracy import final_score

from sklearn.preprocessing import OneHotEncoder

def pca(sFileTrai, sFileTest, iComponents):
    """
    Performs PCA 
    
    Keyword arguments: 
    - sFileTrai: Path to training ivector.scp file 
    - sFileTest: Path to testing ivector.scp file 
    - iComponents: No of components to perform 
    
    Returns: 
    - vTraiPCA: Training data transformed by PCA for all subject_id 
    - vLTrai: Training labels for all subject_id 
    - vTraiSubjectId: List of the subject_id for training 
    - vTestPCA: Testing data transformed by PCA for all subject_id 
    - vLTest: Test labels for all subject_id 
    - vTestSubjectId: List of the subject_id for test
    """
    dIvecTrai = { key:mat for key,mat in kaldi_io.read_vec_flt_scp(sFileTrai) }
    vTrai= pd.DataFrame((list(dIvecTrai.values())))
    # Takes the last character in the filename as it is the label 
    vLTrai = np.array([x[-1] for x in np.array(list(dIvecTrai.keys()))])
    
    
    pca = PCA(n_components=iComponents, svd_solver='randomized', whiten=True)
    pca.fit(vTrai)
        
    vTraiPCA=pca.transform(vTrai)

    # FIXME : For realPD, we need more than -5 (CIS-PD subject_id is 4 characters long)
    # FIXME REAL-PD it's not only int
    vTraiSubjectId = np.array(([int(x[-5:-1]) for x in np.array(list(dIvecTrai.keys()))]))

    dIvecTest = { key:mat for key,mat in kaldi_io.read_vec_flt_scp(sFileTest) }
    vTest=np.array(list(dIvecTest.values()), dtype=float)
    vLTest=np.array([int(x[-1]) for x in np.array(list(dIvecTest.keys()))])
    vTestSubjectId = np.array([int(x[-5:-1]) for x in np.array(list(dIvecTest.keys()))])
    vTestMeasurementId =  np.array([x[-42:-6] for x in np.array(list(dIvecTest.keys()))])
    # Builds a list of the measurement_id to use for the testing_data subset
    sPatternMeasurementId = r'(?<=trai_)[a-z\-0-9]+(?=[_])'
    #vTestMeasurementId = np.array([re.findall(sPatternMeasurementId, fileName)[0] for fileName in np.array(list(dIvecTest.keys()))])

    # Get the measurement_id here
    vTestPCA=pca.transform(vTest)

    if isinstance(iComponents, str):
        iComponents=int(iComponents)
        
    return vTraiPCA, vLTrai, vTraiSubjectId, vTestPCA, vLTest, vTestSubjectId, vTestMeasurementId

def pca_knn_bpd2(sFileTrai, sFileTest, sOut, iComponents, iNeighbors=None, sKernel=None, fCValue=None, fEpsilon=None):
    """
    Performs PCA, then KNN and dumps the results in a pickle file 
    
    Keyword arguments: 
    - sFileTrai: TODO
    - sFileTest: TODO
    - sOut: TODO
    - iComponents: TODO
    - iNeighbors: TODO
    - sFileTestSubset: Path to the real files for testing subset 
    """
    if isinstance(iComponents, str):
        iComponents=int(iComponents)
    if isinstance(iNeighbors, str):
        iNeighbors=int(iNeighbors)
    if isinstance(fCValue, str):
        fCValue=float(fCValue)
    if isinstance(fEpsilon, str):
        fEpsilon=float(fEpsilon)
        
    vTraiPCA, vLTrai, \
    vTraiSubjectId, vTestPCA, \
    vLTest, vTestSubjectId, \
    vTestMeasurementId = pca(sFileTrai, sFileTest, iComponents)

    # To compute the mean accuracy
    glob_trai_pred=[]
    glob_test_pred=[]
    glob_trai_true=[]
    glob_test_true=[]
    glob_test_mesID=[]
    
    
        
    lScoreTrai = []
    lScoreTest = []
    vTraiPCA_all = np.array([]).reshape(0,iComponents+len(np.unique(vTraiSubjectId)))
    vLTrai_all = np.array([]).reshape(0,)
    vTestPCA_all = np.array([]).reshape(0,iComponents+len(np.unique(vTraiSubjectId)))
    vLTest_all = np.array([]).reshape(0,)

    uniq_subjectid = np.unique(vTraiSubjectId)
    enc = OneHotEncoder(handle_unknown='ignore').fit(uniq_subjectid.reshape(-1,1))
    for subject_id in np.unique(vTraiSubjectId):
        print('----- ' + str(subject_id) + '----- ')
        if iNeighbors is not None:
            print('Using KNN') 
            knn = KNeighborsClassifier(n_neighbors=iNeighbors)
            sScoreType = 'accuracy'
        else: 
            print('Using SVR')
            #knn = LinearSVR(epsilon=fEpsilon) 
            knn = SVR(kernel=sKernel, C=fCValue, epsilon=fEpsilon, gamma='auto')
            sScoreType = 'R2'

        # Filter vTraiPCA and vLTraiPCA for one subject_id
        indices_subject_id = np.where(vTraiSubjectId == subject_id) 
        print('vTraiPCA.shape ', vTraiPCA.shape)
        vTraiPCA_subjectid = vTraiPCA[indices_subject_id]
        vLTrai_subjectid = vLTrai[indices_subject_id]
        vLTrai_subjectid = vLTrai_subjectid.astype(int)
        print('vTraiPCA_subjectid.shape : ', vTraiPCA_subjectid.shape)
        vMeanTrai_subjectid = np.mean(vTraiPCA_subjectid,axis=0)
        vMNTraiPCA_subjectid = vTraiPCA_subjectid - vMeanTrai_subjectid
        
        onehot_subjectid = enc.transform([[subject_id]]).toarray()
        onehotTrai_subjectid = np.repeat(onehot_subjectid,vMNTraiPCA_subjectid.shape[0],axis=0)
        
        vMNTraiPCA_subjectid = np.concatenate((vMNTraiPCA_subjectid,onehotTrai_subjectid),axis=1)
        print('vMNTraiPCA_subjectid shape : ', vMNTraiPCA_subjectid.shape)
        
        # Filter vTestPCA and vLTestPCA for one subject_id
        indices_subject_id = np.where(vTestSubjectId == subject_id)
        vTestPCA_subjectid = vTestPCA[indices_subject_id]
        vLTest_subjectid = vLTest[indices_subject_id]
        vLTest_subjectid = vLTest_subjectid.astype(int)
        lTestMeasId_subjectid = vTestMeasurementId[indices_subject_id] # measID per participant
        
        vMNTestPCA_subjectid = vTestPCA_subjectid - vMeanTrai_subjectid
        
        onehotTest_subjectid = np.repeat(onehot_subjectid,vMNTestPCA_subjectid.shape[0],axis=0)
        vMNTestPCA_subjectid = np.concatenate((vMNTestPCA_subjectid,onehotTest_subjectid),axis=1)
        
        vTraiPCA_all = np.concatenate((vTraiPCA_all,vMNTraiPCA_subjectid),axis=0)
        vLTrai_all = np.concatenate((vLTrai_all,vLTrai_subjectid),axis=0)
        vTestPCA_all = np.concatenate((vTestPCA_all,vMNTestPCA_subjectid),axis=0)
        vLTest_all = np.concatenate((vLTest_all,vLTest_subjectid),axis=0)
        
        # We train the KNN only on the data for one subject_id 
    N = vTraiPCA_all.shape[0]
    ind = np.random.permutation(N)
    vTraiPCA_all = vTraiPCA_all[ind,:]
    vLTrai_all = vLTrai_all[ind]
    
    knn.fit(vTraiPCA_all, vLTrai_all)
    lScoreTrai.append(knn.score(vTraiPCA_all, vLTrai_all))
    lScoreTest.append(knn.score(vTestPCA_all, vLTest_all))
    
    print('Training '+sScoreType+': ', knn.score(vTraiPCA_all, vLTrai_all))
    print('Testing '+sScoreType+': ', knn.score(vTestPCA_all, vLTest_all))

    # Predicting on the training and test data
    predictionsTrai = knn.predict(vTraiPCA_all)
    predictions = knn.predict(vTestPCA_all)

    # Computing the accuracy
    glob_trai_pred=predictionsTrai
    glob_test_pred=predictions
    glob_trai_true=vLTrai_all
    glob_test_true=vLTest_all
    glob_test_mesID=vTestMeasurementId
#     glob_test_mesID.append(lTestMeasId_subjectid)

    # Building a list of the MSEk
    # To compute the final score as per the challenge
    mse_training_per_subjectid=[]
    mse_test_per_subjectid=[]
    train_nb_files_per_subjectid=[]
    test_nb_files_per_subjectid=[]
    for subject_id in np.unique(vTraiSubjectId):
        print('----- ' + str(subject_id) + '----- ')
        indices_subject_id = np.where(vTraiSubjectId == subject_id) 
        vTraiPCA_subjectid = vTraiPCA[indices_subject_id]
        vLTrai_subjectid = vLTrai[indices_subject_id]
        vLTrai_subjectid = vLTrai_subjectid.astype(int)
        
        vMeanTrai_subjectid = np.mean(vTraiPCA_subjectid,axis=0)
        vMNTraiPCA_subjectid = vTraiPCA_subjectid - vMeanTrai_subjectid
        
        onehot_subjectid = enc.transform([[subject_id]]).toarray()
        onehotTrai_subjectid = np.repeat(onehot_subjectid,vMNTraiPCA_subjectid.shape[0],axis=0)
        vMNTraiPCA_subjectid = np.concatenate((vMNTraiPCA_subjectid,onehotTrai_subjectid),axis=1)
        
        # Filter vTestPCA and vLTestPCA for one subject_id
        indices_subject_id = np.where(vTestSubjectId == subject_id)
        vTestPCA_subjectid = vTestPCA[indices_subject_id]
        vLTest_subjectid = vLTest[indices_subject_id]
        vLTest_subjectid = vLTest_subjectid.astype(int)
        lTestMeasId_subjectid = vTestMeasurementId[indices_subject_id] # measID per participant
        
        vMNTestPCA_subjectid = vTestPCA_subjectid - vMeanTrai_subjectid
        
        onehotTest_subjectid = np.repeat(onehot_subjectid,vMNTestPCA_subjectid.shape[0],axis=0)
        vMNTestPCA_subjectid = np.concatenate((vMNTestPCA_subjectid,onehotTest_subjectid),axis=1)
        
        predictionsTrai = knn.predict(vMNTraiPCA_subjectid)
        predictions = knn.predict(vMNTestPCA_subjectid) 
        
        mse_training_per_subjectid = np.append(mse_training_per_subjectid,
                                               (mean_squared_error(vLTrai_subjectid, predictionsTrai)))
        mse_test_per_subjectid = np.append(mse_test_per_subjectid,
                                            (mean_squared_error(vLTest_subjectid, predictions)))
        train_nb_files_per_subjectid.append(len(vLTrai_subjectid))
        test_nb_files_per_subjectid.append(len(vLTest_subjectid))
        
        
        print('mean_squared_error train for subject id: ', mean_squared_error(vLTrai_subjectid, predictionsTrai)) 
        print('mean_squared_error test for subject id: ', mean_squared_error(vLTest_subjectid, predictions)) 
    #import pdb; pdb.set_trace()
    final_score(mse_training_per_subjectid.tolist(),train_nb_files_per_subjectid, 'Training ')
    
    # print('True Labels Training:')
    # print(glob_trai_true)
    # print('#')
    # print('#')
    # print('Predicted Labels Training:')
    # print(glob_trai_pred)
    # print('#')
    # print('#')
    # print('True Labels Testing:')
    # print(glob_test_true)
    # print('#')
    # print('#')
    # print('Predicted Labels Testing:')
    # print(glob_test_pred)
    # print('#')
    # print('#')
    print('sOut : ', sOut)
    if not  os.path.isdir(sOut):
        os.mkdir(sOut)
    
    if iNeighbors is not None:
        sObjname='objs_'+str(iComponents)+'_k_'+str(iNeighbors)+'.pkl'
    else: 
        sObjname='objs_everyone_'+str(iComponents)+'_kernel_'+str(sKernel)+ \
                                          '_c_'+str(fCValue)+ \
                                          '_eps_'+str(fEpsilon)+'.pkl'
                
    with open(os.path.join(sOut,sObjname), 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([glob_trai_pred,glob_trai_true,glob_test_pred,glob_test_true, \
                    mse_training_per_subjectid,mse_test_per_subjectid, \
                     train_nb_files_per_subjectid,test_nb_files_per_subjectid, glob_test_mesID], f)
        
    print('----- GLOBAL -----')
    print('PCAComponents: {}'.format((iComponents)))
    if iNeighbors is not None:
        print('Global training accuracy: {}'.format((glob_trai_true == glob_trai_pred).mean()))
        print('Global testing accuracy: {}'.format((glob_test_true == glob_test_pred).mean()))
        print('iNeighbors: {}'.format((iNeighbors)))
    else:
        print('Global training R2: {}'.format((sum(lScoreTrai) / len(lScoreTrai))))
        print('Global testing R2: {}'.format((sum(lScoreTest) / len(lScoreTest))))


if __name__ == "__main__":
    # Usage example : 
    # sFileTrai = '/export/c08/lmorove1/kaldi/egs/beatPDivec/v1/exp/ivectors_Training_Fold0/ivector.scp'
    # sFileTest = '/export/c08/lmorove1/kaldi/egs/beatPDivec/v1/exp/ivectors_Testing_Fold0/ivector.scp'
    # iComponents = 50 
    
    parser=argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars='@',
        description='Trains and Tests KNN.')

    parser.add_argument('--input-trai',dest='sFileTrai', required=True)
    parser.add_argument('--input-test',dest='sFileTest', required=True)
    parser.add_argument('--output-file', dest='sOut', required=True)
    parser.add_argument('--iComponents', dest='iComponents', required=True)
    # KNN
    parser.add_argument('--iNeighbors', dest='iNeighbors')
    
    # SVR 
    parser.add_argument('--sKernel', dest='sKernel')
    parser.add_argument('--fCValue', dest='fCValue')
    parser.add_argument('--fEpsilon', dest='fEpsilon')

    args=parser.parse_args()
    
    pca_knn_bpd2(**vars(args))
                                    
