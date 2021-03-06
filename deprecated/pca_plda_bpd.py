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

# MSE 
from sklearn.metrics import mean_squared_error

def pca(sFileTrai, sFileTest, iComponents):
    """
    Performs PCA 
    
    Keyword arguments: 
    - sFileTrai: TODO
    - sFileTest: TODO
    - iComponents: No of components to perform 
    
    Returns: 
    - vTraiPCA: TODO
    - vLTrai: TODO
    - vTraiSubjectId: List of the subject_id for training 
    - vTestPCA: TODO
    - vLTest: TODO
    - vTestSubjectId: List of the subject_id for test
    """
    dIvecTrai = { key:mat for key,mat in kaldi_io.read_vec_flt_scp(sFileTrai) }
    vTrai= pd.DataFrame((list(dIvecTrai.values())))
    vLTrai = np.array([x[-1] for x in np.array(list(dIvecTrai.keys()))])
    
    # FIXME : For realPD, we need more than -5 (CIS-PD subject_id is 4 characters long)
    # FIXME REAL-PD it's not only int
    vTraiSubjectId = np.array(([int(x[-5:-1]) for x in np.array(list(dIvecTrai.keys()))]))
    
    dIvecTest = { key:mat for key,mat in kaldi_io.read_vec_flt_scp(sFileTest) }
    vTest=np.array(list(dIvecTest.values()), dtype=float)
    vLTest=np.array([int(x[-1]) for x in np.array(list(dIvecTest.keys()))])
    vTestSubjectId = np.array([int(x[-5:-1]) for x in np.array(list(dIvecTest.keys()))])

    #iComponents=60;
    
    if isinstance(iComponents, str):
        iComponents=int(iComponents)
        
    pca = PCA(n_components=iComponents, svd_solver='randomized', whiten=True)
    pca.fit(vTrai)
        
    vTraiPCA=pca.transform(vTrai)
    vTestPCA=pca.transform(vTest)

    return vTraiPCA, vLTrai, vTraiSubjectId, vTestPCA, vLTest, vTestSubjectId

def pca_plda_bpd(sFileTrai, sFileTest, sOut, iComponents):
    """
    Performs PCA, then PLDA and dumps the results in a pickle file 
    
    Keyword arguments: 
    - sFileTrai: TODO
    - sFileTest: TODO
    - sOut: TODO
    - iComponents: TODO
    - iNeighbors: TODO
    """
    vTraiPCA, vLTrai, vTraiSubjectId, vTestPCA, vLTest, vTestSubjectId = pca(sFileTrai, sFileTest, iComponents)

    # To compute the mean accuracy
    glob_trai_pred=[]
    glob_test_pred=[]
    glob_trai_true=[]
    glob_test_true=[]
    
    # To compute the final score as per the challenge
    mse_training_per_subjectid=[]
    mse_test_per_subjectid=[]
    train_nb_files_per_subjectid=[]
    test_nb_files_per_subjectid=[]

    for subject_id in np.unique(vTraiSubjectId):
        print('----- ' + str(subject_id) + '----- ')

        # Filter vTraiPCA and vLTraiPCA for one subject_id
        indices_subject_id = np.where(vTraiSubjectId == subject_id)
        vTraiPCA_subjectid = vTraiPCA[indices_subject_id]
        vLTrai_subjectid = vLTrai[indices_subject_id]

        print('Previous shapes:')
        print(vTraiPCA_subjectid.shape)
        print(vLTrai_subjectid.shape)

        # remove single instances
        for x in np.unique(vLTrai_subjectid):
            indices_subject_id = np.where(vLTrai_subjectid == x)
            siz=indices_subject_id[0].shape
            if siz[0] < 2:
                print('Deleting class: ')
                print(x)
                print(len(indices_subject_id))
                print(indices_subject_id)
                vLTrai_subjectid=np.delete(vLTrai_subjectid,indices_subject_id)
                vTraiPCA_subjectid=np.delete(vTraiPCA_subjectid,indices_subject_id,axis=0)

        print('Shapes after removing classes:')
        print(vTraiPCA_subjectid.shape)
        print(vLTrai_subjectid.shape)
        vLTrai_subjectid=vLTrai_subjectid.astype(int)

        # Filter vTestPCA and vLTestPCA for one subject_id
        indices_subject_id = np.where(vTestSubjectId == subject_id)
        vTestPCA_subjectid = vTestPCA[indices_subject_id]
        vLTest_subjectid = vLTest[indices_subject_id]
        vLTest_subjectid=vLTest_subjectid.astype(int)

            # PLDA
        classifier = plda.Classifier()
        classifier.fit_model(vTraiPCA_subjectid, vLTrai_subjectid )
            
        predictions, log_p_predictions = classifier.predict(vTestPCA_subjectid)
        predictionsTrai, log_p_predictionsTrai = classifier.predict(vTraiPCA_subjectid)
        #print(predictions)
        print('Accuracy: {}'.format((vLTest_subjectid == predictions).mean()))
        glob_trai_pred=np.append(glob_trai_pred,predictionsTrai,axis=0)
        glob_test_pred=np.append(glob_test_pred,predictions,axis=0)
        glob_trai_true=np.append(glob_trai_true,vLTrai_subjectid,axis=0)
        glob_test_true=np.append(glob_test_true,vLTest_subjectid,axis=0)

        # Building a list of the MSEk
        mse_training_per_subjectid = np.append(mse_training_per_subjectid,
                                                   (mean_squared_error(vLTrai_subjectid, predictionsTrai) / len(vLTrai_subjectid)))
        mse_test_per_subjectid = np.append(mse_test_per_subjectid,
                                           (mean_squared_error(vLTest_subjectid, predictions) / len(vLTest_subjectid)))
        train_nb_files_per_subjectid.append(len(vLTrai_subjectid))
        test_nb_files_per_subjectid.append(len(vLTest_subjectid))

    print('Global training accuracy: {}'.format((glob_trai_true == glob_trai_pred).mean()))
    print('Global testing accuracy: {}'.format((glob_test_true == glob_test_pred).mean()))
    print('PCAComponents: {}'.format((iComponents)))
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

    if not  os.path.isdir(sOut):
            os.mkdir(sOut)
    sObjname='objs_'+str(iComponents)+'.pkl'
    with open(os.path.join(sOut,sObjname), 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([glob_trai_pred,glob_trai_true,glob_test_pred,glob_test_true, \
                    mse_training_per_subjectid,mse_test_per_subjectid, \
                    train_nb_files_per_subjectid,test_nb_files_per_subjectid], f)


if __name__ == "__main__":

    parser=argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars='@',
        description='Trains and Tests PCA-PLDA.')

    parser.add_argument('--input-trai',dest='sFileTrai', required=True)
    parser.add_argument('--input-test',dest='sFileTest', required=True)
    parser.add_argument('--output-file', dest='sOut', required=True)
    parser.add_argument('--iComponents', dest='iComponents', required=True)
    
    args=parser.parse_args()
    
    pca_plda_bpd(**vars(args))
                                    
