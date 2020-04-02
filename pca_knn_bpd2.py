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
    
    # To compute the final score as per the challenge
    mse_training_per_subjectid=[]
    mse_test_per_subjectid=[]
    train_nb_files_per_subjectid=[]
    test_nb_files_per_subjectid=[]
    
        
    lScoreTrai = []
    lScoreTest = []
    for subject_id in np.unique(vTraiSubjectId):
        print('----- ' + str(subject_id) + '----- ')
        if iNeighbors is not None:
            print('Using KNN') 
            knn = KNeighborsClassifier(n_neighbors=iNeighbors)
            sScoreType = 'accuracy'
        else: 
            print('Using SVR')
            knn = SVR(kernel=sKernel, C=fCValue, epsilon=fEpsilon, gamma='auto')
            sScoreType = 'R2'

        # Filter vTraiPCA and vLTraiPCA for one subject_id
        indices_subject_id = np.where(vTraiSubjectId == subject_id) 
        vTraiPCA_subjectid = vTraiPCA[indices_subject_id]
        vLTrai_subjectid = vLTrai[indices_subject_id]
        vLTrai_subjectid = vLTrai_subjectid.astype(int)

        # Filter vTestPCA and vLTestPCA for one subject_id
        indices_subject_id = np.where(vTestSubjectId == subject_id)
        vTestPCA_subjectid = vTestPCA[indices_subject_id]
        vLTest_subjectid = vLTest[indices_subject_id]
        vLTest_subjectid = vLTest_subjectid.astype(int)
        lTestMeasId_subjectid = vTestMeasurementId[indices_subject_id] # measID per participant
        
        # We train the KNN only on the data for one subject_id 
        knn.fit(vTraiPCA_subjectid, vLTrai_subjectid)
        lScoreTrai.append(knn.score(vTraiPCA_subjectid, vLTrai_subjectid))
        lScoreTest.append(knn.score(vTestPCA_subjectid, vLTest_subjectid))
       
        print('Training '+sScoreType+': ', knn.score(vTraiPCA_subjectid, vLTrai_subjectid))
        print('Testing '+sScoreType+': ', knn.score(vTestPCA_subjectid, vLTest_subjectid))
        
        # Predicting on the training and test data
        predictionsTrai = knn.predict(vTraiPCA_subjectid)
        predictions = knn.predict(vTestPCA_subjectid)

        # Computing the accuracy
        glob_trai_pred=np.append(glob_trai_pred,predictionsTrai,axis=0)
        glob_test_pred=np.append(glob_test_pred,predictions,axis=0)
        glob_trai_true=np.append(glob_trai_true,vLTrai_subjectid,axis=0)
        glob_test_true=np.append(glob_test_true,vLTest_subjectid,axis=0)
        glob_test_mesID.append(lTestMeasId_subjectid)

        # Building a list of the MSEk
        print('Training MSE : ', mean_squared_error(vLTrai_subjectid, predictionsTrai))
        print('Testing MSE : ', mean_squared_error(vLTest_subjectid, predictions))
        mse_training_per_subjectid = np.append(mse_training_per_subjectid,
                                               (mean_squared_error(vLTrai_subjectid, predictionsTrai)))
        mse_test_per_subjectid = np.append(mse_test_per_subjectid,
                                            (mean_squared_error(vLTest_subjectid, predictions)))
        train_nb_files_per_subjectid.append(len(vLTrai_subjectid))
        test_nb_files_per_subjectid.append(len(vLTest_subjectid))
        
#         print('mean_squared_error train for subject id: ', mean_squared_error(vLTrai_subjectid, np.repeat(np.mean(vLTrai_subjectid), len(vLTrai_subjectid)))) 
#         print('mean_squared_error test for subject id: ', mean_squared_error(vLTest_subjectid, np.repeat(np.mean(vLTest_subjectid), len(vLTest_subjectid)))) 
 
    
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
    
    if iNeighbors is not None:
        sObjname='objs_'+str(iComponents)+'_k_'+str(iNeighbors)+'.pkl'
    else: 
        sObjname='objs_'+str(iComponents)+'_kernel_'+str(sKernel)+ \
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
                                    
