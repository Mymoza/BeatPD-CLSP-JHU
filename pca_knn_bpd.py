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

# KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition


def pca(sFileTrai, sFileTest, iComponents):
    # os.chdir("/export/c10/lmorove1/kaldi/egs/xVecPD/v2_NeuroConc16k")
    # sFileTrai="/export/c10/lmorove1/kaldi/egs/xVecPD/v2_NeuroConc16k/exp/sre18nn/xvectors_Training_Fold1/xvector.scp"
    # sOut = Folder where to store results
    # iComponents = nb of classes (nb of measurements_id? or nb of subject_ids?)
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

    #print('after transform : ', vTraiPCA.shape)
    #print('after transform : ', vTestPCA.shape)
    return vTraiPCA, vLTrai, vTraiSubjectId, vTestPCA, vLTest, vTestSubjectId

def pca_plda_bpd(sFileTrai, sFileTest, sOut, iComponents):

    vTraiPCA, vLTrai, vTraiSubjectId, vTestPCA, vLTest, vTestSubjectId = pca(sFileTrai, sFileTest, iComponents)

    # To compute the mean accuracy
    glob_trai_pred=[]
    glob_test_pred=[]
    glob_trai_true=[]
    glob_test_true=[]

    for subject_id in np.unique(vTraiSubjectId):
        # GridSearchCV
    #     parameters = {'n_neighbors':[1,3,6,10,15]}

        k_range = list(range(1, 15))
    #     param_grid = dict(n_neighbors=k_range)
    #     print(param_grid)

        # instantiate the grid
    #     grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')

        # fit the grid with data
    #     grid.fit(vTraiPCA_subjectid, vLTrai_subjectid)

        # view the complete results (list of named tuples)
    #     print(grid.cv_results_['mean_test_score'])

        print('----- ' + str(subject_id) + '----- ')
        knn = KNeighborsClassifier(n_neighbors=6)

        # Filter vTraiPCA and vLTraiPCA for one subject_id
        indices_subject_id = np.where(vTraiSubjectId == subject_id) # HAPPY
        vTraiPCA_subjectid = vTraiPCA[indices_subject_id]
        vLTrai_subjectid = vLTrai[indices_subject_id]

        # Filter vTestPCA and vLTestPCA for one subject_id
        indices_subject_id = np.where(vTestSubjectId == subject_id)
        vTestPCA_subjectid = vTestPCA[indices_subject_id]
        vLTest_subjectid = vLTest[indices_subject_id]

        knn.fit(vTraiPCA_subjectid, vLTrai_subjectid)

    #     y_labels = knn.predict(vTraiPCA_subjectid)
    #     display(y_labels)

        print('Training accuracy: ', knn.score(vTraiPCA_subjectid, vLTrai_subjectid))
        print('Testing accuracy: ', knn.score(vTestPCA_subjectid, vLTest_subjectid))

        # Predicting on the training and test data 
        predictionsTrai = knn.predict(vTraiPCA_subjectid)
        predictions = knn.predict(vTestPCA_subjectid)

        # Converting all strings to int for MSE 
        vLTrai_subjectid = [int(i) for i in vLTrai_subjectid]
        predictionsTrai = [int(i) for i in predictionsTrai]
        vLTest_subjectid = [int(i) for i in vLTest_subjectid]
        predictions = [int(i) for i in predictions]

        # Computing the accuracy 
        glob_trai_pred=np.append(glob_trai_pred,predictionsTrai,axis=0)
        glob_test_pred=np.append(glob_test_pred,predictions,axis=0)
        glob_trai_true=np.append(glob_trai_true,vLTrai_subjectid,axis=0)
        glob_test_true=np.append(glob_test_true,vLTest_subjectid,axis=0)

        a = [a - b for a, b in zip(vLTrai_subjectid, predictionsTrai)]
        print('abs substract : ', np.abs(a))

        # Building a list of the MSEk 
        mse_training_per_subjectid = np.append(mse_training_per_subjectid,
                                               (mean_squared_error(vLTrai_subjectid, predictionsTrai) /  len(vLTrai_subjectid)))
        mse_test_per_subjectid = np.append(mse_test_per_subjectid,
                                            (mean_squared_error(vLTest_subjectid, predictions) / len(vLTest_subjectid)))
        train_nb_files_per_subjectid.append(len(vLTrai_subjectid))
        test_nb_files_per_subjectid.append(len(vLTest_subjectid))

    #     do_confusion_matrix(vLTest_subjectid, predictions)

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
        pickle.dump([glob_trai_pred,glob_trai_true,glob_test_pred,glob_test_true], f)

def pca_plda_bpd(sFileTrai, sFileTest, sOut, iComponents):
    

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
    
    args=parser.parse_args()
    
    pca_plda_bpd(**vars(args))
                                    
