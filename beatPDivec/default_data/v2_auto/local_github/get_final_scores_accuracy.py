#!/usr/bin/env python3.7
import pickle
import re 
import numpy as np
import glob
import argparse
from sklearn.metrics import mean_squared_error
from math import sqrt 
import pandas as pd 
import os
from sklearn.metrics import r2_score
import pprint

def final_score(mse_per_subjectid, nb_files_per_subject_id, training_or_test=''):
    """
    Compute the final score for the challenge given the arguments 
    
    Keyword arguments:
    - mse_per_subjectid: list of the mse per subject_id 
    - nb_files_per_subject_id: list of the number of files per subject_id 
    - training_or_test: string just for the purpose of printing the result 
    """
    numerator = np.sum([nb_file * mse for nb_file, mse in zip(np.sqrt(nb_files_per_subject_id), mse_per_subjectid)])
    denominator = np.sum(np.sqrt(nb_files_per_subject_id))
    #FIXME : Refactor so it's not printing by default 
    print(training_or_test+'Final score : ', np.divide(numerator, denominator))
    return np.divide(numerator, denominator)

def get_final_score(vPredictions, vParID, vTrueLabels):
    """
    Compute the final score from the challenge and print the result
    
    Keyword arguments: 
    - vPredictions: Numpy array containing the predictions 
    - VParID: list containing the subject_id 
    - vTrueLabels: list containing the true labels 
    """
    mse_per_subjectId = []
    nb_files_per_subjectId = []
    
    for subject_id in np.unique(vParID):
#         print('--- SUBJECT ID ', subject_id, '---')
        
        vSubjectId = (vParID == subject_id)

        vPredictions_subjectId = vPredictions[vSubjectId]

        vTrueLabels_subjectId = np.array(vTrueLabels)[vSubjectId]
        mse_per_subjectId.append(mean_squared_error(vTrueLabels_subjectId, vPredictions_subjectId))
        nb_files_per_subjectId.append(len(vPredictions_subjectId))
        
        print('MSE : ', mean_squared_error(vTrueLabels_subjectId, vPredictions_subjectId))
    
    print('--- MSEscore ---')
    final_score(mse_per_subjectId, nb_files_per_subjectId)
    
def find_components_neighbors(lObjsFiles, bKnn, bSVR, bEveryoneSVR): 
    lComponents = [] 
    
    #KNN
    lNeighbors = []
    
    # SVR 
    lKernel = []
    lCValue = [] 
    lEpsilon = []
    
    # Get a list of all no of components 
    for objsFile in lObjsFiles:
        # Get a list of all iComponents
        if bEveryoneSVR:
            sPatternComponents = r'(?<=objs_everyone_)\d+(?=[_|.])'
        else: 
            sPatternComponents = r'(?<=objs_)\d+(?=[_|.])'
        # Only add it to the list if it's not already there 
        noToAdd = re.findall(sPatternComponents, objsFile)[0]
        lComponents.append(noToAdd) if noToAdd not in lComponents else lComponents
        
        if bKnn:
            # Get a list of all iNeighbors
            sPatternNeighbors = r'(?<=k_)\d+'
            print('no To Add : ', re.findall(sPatternNeighbors, objsFile))
            noToAdd = re.findall(sPatternNeighbors, objsFile)[0]
            lNeighbors.append(noToAdd) if noToAdd not in lNeighbors else lNeighbors
        elif bSVR or bEveryoneSVR:
            sPatternKernel = r'(?<=kernel_).*(?=_c)'
            kernelToAdd = re.findall(sPatternKernel, objsFile)[0]
            lKernel.append(kernelToAdd) if kernelToAdd not in lKernel else lKernel

            sPatternC = r'(?<=_c_).*(?=_eps)'
            cToAdd = re.findall(sPatternC, objsFile)[0]
            lCValue.append(cToAdd) if cToAdd not in lCValue else lCValue
            
            sPatternEpsilon = r'(?<=eps_).*(?=\.)'
            epsToAdd = re.findall(sPatternEpsilon, objsFile)[0]
            lEpsilon.append(epsToAdd) if epsToAdd not in lEpsilon else lEpsilon
            
        else:
            # Quickfix to ignore the Neighbors loop if an algorithm other than KNN is used 
            lNeighbors = [1]
    
    print('Components found : ', lComponents)
    if bKnn:
        print('Neighbors found : ', lNeighbors)
    elif bSVR or bEveryoneSVR:
        print('Kernels found : ', lKernel)
        print('C Values found : ', lCValue) 
        print('Epsilon found : ', lEpsilon) 

        return lComponents, lKernel, lCValue, lEpsilon
        
    return lNeighbors, lComponents
    
def find_res_folders(sFilePath, bKnn, bSVR, bEveryoneSVR):
    """
    Keyword arguments: 
    - sFilePath: Path to where the res* folders are. 
    - bKnn: Flag to say if the KNN algorithm is used to go through neighbors combination 
    
    Returns:
    - lResxFolders: Path to the folders 
      example: ['/home/sjoshi/codes/python/BeatPD/code/resxVecFold1', 
                '/home/sjoshi/codes/python/BeatPD/code/resxVecFold4',
                '/home/sjoshi/codes/python/BeatPD/code/resxVecFold0',
                '/home/sjoshi/codes/python/BeatPD/code/resxVecFold3',
                '/home/sjoshi/codes/python/BeatPD/code/resxVecFold2'] 
      
    - lObjsFiles: 
       example: ['/home/sjoshi/codes/python/BeatPD/code/resxVecFold2/objs_50.pkl']
    """
    # Building the list of folders we have to open 
    if bKnn:
        sFolderNamePattern = "resiVecKNN_Fold\d"
    elif bSVR: 
        sFolderNamePattern = "resiVecSVR_Fold\d"
    elif bEveryoneSVR:
        sFolderNamePattern = "resiVecEveryoneSVR_Fold\d"
    else:
        sFolderNamePattern = "resx*"
    
    print('Looking for folder : ', sFolderNamePattern)
    lResxFolders = [sFilePath+f for f in os.listdir(sFilePath) if re.search(sFolderNamePattern, f)]
    sPatternFold = '(?<=[Ff]old)\d+'

    # Get a list of all files starting with objs
    lObjsFiles = []
    for fold_folder in lResxFolders:
        if bKnn:
            lObjsFiles.extend(glob.glob(fold_folder + "/objs*k_*"))
        elif bSVR: 
            lObjsFiles.extend(glob.glob(fold_folder + "/objs*kernel_*"))
        elif bEveryoneSVR: 
            lObjsFiles.extend(glob.glob(fold_folder + "/objs_everyone_*kernel_*"))

    print('lResxFolders : ', lResxFolders)
    print('lObjsFiles : ', lObjsFiles)
    return lResxFolders, lObjsFiles

def get_all_folds(lResxFolders, sFileName):
    """
    Returns all the folds from the provided list of folders (lResxFolders) and that have a certain 
    hyperparameters configuration provided in sFileName 
    
    Keyword arguments:
    - lResxFolders: TODO 
    - sFileName: TODO 
    """
    # To compute mean accuracy accross all folds
    allfolds_glob_trai_pred = []
    allfolds_glob_trai_true = []
    allfolds_glob_test_pred = []
    allfolds_glob_test_true = [] 

    # To compute final score across all folds 
    allfolds_mse_training_per_subjectid = []
    allfolds_mse_test_per_subjectid = []
    allfolds_train_nb_files_per_subjectid = []
    allfolds_test_nb_files_per_subjectid = []
            
    for fold_folder in lResxFolders:
#         print(fold_folder)
#         print('Filename : ', fold_folder+sFileName)
        # Retrieve DataFrames from Pickle file 
        [glob_trai_pred,glob_trai_true, \
         glob_test_pred,glob_test_true, \
         mse_training_per_subjectid, \
         mse_test_per_subjectid, \
         train_nb_files_per_subjectid, \
         test_nb_files_per_subjectid, \
         vTestMeasurementId] = pickle.load(open(fold_folder+sFileName, "rb" ) )

        # Build the DataFrames for all folds 
        allfolds_glob_trai_pred = np.append(allfolds_glob_trai_pred, glob_trai_pred)
        allfolds_glob_trai_true = np.append(allfolds_glob_trai_true, glob_trai_true)
        allfolds_glob_test_pred = np.append(allfolds_glob_test_pred, glob_test_pred)
        allfolds_glob_test_true = np.append(allfolds_glob_test_true, glob_test_true)

        allfolds_mse_training_per_subjectid = np.append(allfolds_mse_training_per_subjectid, 
                                                        mse_training_per_subjectid)
        allfolds_mse_test_per_subjectid = np.append(allfolds_mse_test_per_subjectid,
                                                   mse_test_per_subjectid)
        allfolds_train_nb_files_per_subjectid = np.append(allfolds_train_nb_files_per_subjectid, 
                                                         train_nb_files_per_subjectid)
        allfolds_test_nb_files_per_subjectid = np.append(allfolds_test_nb_files_per_subjectid, 
                                                        test_nb_files_per_subjectid)
    
    return allfolds_glob_trai_pred, \
           allfolds_glob_trai_true, \
           allfolds_glob_test_pred, \
           allfolds_glob_test_true, \
           allfolds_mse_training_per_subjectid, \
           allfolds_mse_test_per_subjectid, \
           allfolds_train_nb_files_per_subjectid, \
           allfolds_test_nb_files_per_subjectid
        
    
def get_final_scores_accuracy(sFilePath, bKnn, bSVR, bEveryoneSVR, bPerSubject):
    """
    Read a pickle file and outputs the global mean accuracy & final score for BeatPD Challenge
    
    Keyword arguments:
    - sFilePath: Path to where the res* folders are. 
    - bKnn: Flag to say if the KNN algorithm is used to go through neighbors combination
    """
    
    lResxFolders, lObjsFiles = find_res_folders(sFilePath, bKnn, bSVR, bEveryoneSVR)
    lNeighbors, lComponents = find_components_neighbors(lObjsFiles, bKnn, bSVR, bEveryoneSVR)
    
    # DataFrame which is going to contain the info of the best hyperparameters combination 
    best_result = pd.DataFrame([['TBD', 100,100,100,100]], columns=['Filename',
                                                                    'Global training accuracy',
                                                                    'Global testing accuracy',
                                                                    'Train Final score',
                                                                    'Test Final score'])
    
    for neighbor in lNeighbors:
        (print('------ FOR NEIGHBOR ', neighbor, '------') if bKnn else '')
        for component in lComponents:
            print('---- FOR COMPONENT ', component, '----')
            
            # To compute mean accuracy accross all folds
            allfolds_glob_trai_pred = []
            allfolds_glob_trai_true = []
            allfolds_glob_test_pred = []
            allfolds_glob_test_true = [] 

            # To compute final score across all folds 
            allfolds_mse_training_per_subjectid = []
            allfolds_mse_test_per_subjectid = []
            allfolds_train_nb_files_per_subjectid = []
            allfolds_test_nb_files_per_subjectid = []
            
            if bKnn:
                sFileName = '/objs_'+component+'_k_'+neighbor+'.pkl'
            else:
                sFileName = '/objs_'+component+'.pkl'
            
            #To compute mean accuracy accross all folds
            # To compute final score across all folds
            allfolds_glob_trai_pred, allfolds_glob_trai_true, \
            allfolds_glob_test_pred, \
            allfolds_glob_test_true, \
            allfolds_mse_training_per_subjectid, \
            allfolds_mse_test_per_subjectid, \
            allfolds_train_nb_files_per_subjectid, \
            allfolds_test_nb_files_per_subjectid = get_all_folds(lResxFolders, sFileName)
            
            global_training_accuracy = (allfolds_glob_trai_true == allfolds_glob_trai_pred).mean()
            global_testing_accuracy = (allfolds_glob_test_true == allfolds_glob_test_pred).mean()
            print('Global training accuracy: {}'.format(global_training_accuracy))
            print('Global testing accuracy: {}'.format(global_testing_accuracy))
            
            global_training_final_score = final_score(allfolds_mse_training_per_subjectid,
                        allfolds_train_nb_files_per_subjectid,
                        training_or_test='Train ')
            global_testing_final_score = final_score(allfolds_mse_test_per_subjectid,
                        allfolds_test_nb_files_per_subjectid,
                        training_or_test='Test ')

            if best_result.loc[0,'Test Final score'] > global_testing_final_score:
                best_result.update({'Filename': [sFileName],
                                    'Global training accuracy':[global_training_accuracy],
                                    'Global testing accuracy':[global_testing_accuracy],
                                    'Train Final score':[global_training_final_score],
                                    'Test Final score':[global_testing_final_score]})
                
    
    print('------ GLOBAL WINNER PARAMETERS ------')
    print(best_result.transpose()[0].to_string())
    
    
    # The following section is used to create a csv file to submit to the challenge 
    # To use it, perform pca_knn_bpd() with the training data and the testing data will be the real test subset,
    # not the development one from the folds 
    
    # Loading the file with that obtained the best results
#     lNeighbors, lComponents = find_cogmponents_neighbors([sFilePath+best_result.Filename[0]], bKnn)
    # iComponents = best_result 
#     for neighbor in lNeighbors:
#         (print('------ FOR NEIGHBOR ', neighbor, '------') if bKnn else '')
#         for component in lComponents:
#             print('---- FOR COMPONENT ', component, '----')
            
#             for fold_folder in lResxFolders:
            
#                 [glob_trai_pred,glob_trai_true, \
#                              glob_test_pred,glob_test_true, \
#                              mse_training_per_subjectid, \
#                              mse_test_per_subjectid, \
#                              train_nb_files_per_subjectid, \
#                              test_nb_files_per_subjectid] = pickle.load(open(fold_folder+best_result.Filename[0], "rb" ) )
#                              #vTestMeasurementId] = pickle.load(open(fold_folder+best_result.Filename[0], "rb" ) )

#              # Build the DataFrames for all folds 
#                 allfolds_glob_trai_pred = np.append(allfolds_glob_trai_pred, glob_trai_pred)
#                 allfolds_glob_trai_true = np.append(allfolds_glob_trai_true, glob_trai_true)
#                 allfolds_glob_test_pred = np.append(allfolds_glob_test_pred, glob_test_pred)
# #                 allfolds_glob_test_true = np.append(allfolds_glob_test_true, glob_test_true)

#                 allfolds_mse_training_per_subjectid = np.append(allfolds_mse_training_per_subjectid, 
#                                                                 mse_training_per_subjectid)
# #                 allfolds_mse_test_per_subjectid = np.append(allfolds_mse_test_per_subjectid,
# #                                                            mse_test_per_subjectid)
#                 allfolds_train_nb_files_per_subjectid = np.append(allfolds_train_nb_files_per_subjectid, 
#                                                                  train_nb_files_per_subjectid)
#                 allfolds_test_nb_files_per_subjectid = np.append(allfolds_test_nb_files_per_subjectid, 
#                                                                 test_nb_files_per_subjectid)
    
    # Concatenating the measurement_id and the predictions 
#     pdTestPredictions = pd.concat([pd.DataFrame(vTestMeasurementId),
#                                    pd.DataFrame(allfolds_glob_test_pred)], axis=1)
    
#     pdTestPredictions.to_csv(
#         sOut + 'predictions_'+str(iComponents)+'_k_'+str(iNeighbors)+".csv",
#         index=False,
#         header=["measurement_id","prediction"],
#     )

def get_final_scores_SVR(sFilePath, bKnn, bSVR, bEveryoneSVR, bPerSubject, sDatabase, sSubchallenge):
    """
    Read a pickle file and outputs the global mean accuracy & final score for BeatPD Challenge
    
    Keyword arguments:
    - sFilePath: Path to where the res* folders are. 
    - bKnn: Flag to say if the KNN algorithm is used to go through neighbors combination
    - bEveryoneSVR: TODO
    """
    
    lResxFolders, lObjsFiles = find_res_folders(sFilePath, bKnn, bSVR, bEveryoneSVR)
    lComponents, lKernel, lCValue, lEpsilon = find_components_neighbors(lObjsFiles, bKnn, bSVR, bEveryoneSVR)
    
    # DataFrame which is going to contain the info of the best hyperparameters combination 
    best_result = pd.DataFrame([['TBD', 100,100,100,100]], columns=['Filename',
                                                                    'Global training r2',
                                                                    'Global testing r2',
                                                                    'Train Final score',
                                                                    'Test Final score'])
    for kernel in lKernel:
        print('------ FOR KERNEL ', kernel, '------')
        for c_value in lCValue:
            print('------ FOR C VALUE ', c_value, '------')
            for epsilon in lEpsilon:
                print('------ FOR EPSILON ', epsilon, '------')
                for component in lComponents:
                    print('---- FOR COMPONENT ', component, '----')

                    # To compute mean accuracy accross all folds
                    allfolds_glob_trai_pred = []
                    allfolds_glob_trai_true = []
                    allfolds_glob_test_pred = []
                    allfolds_glob_test_true = [] 

                    # To compute final score across all folds 
                    allfolds_mse_training_per_subjectid = []
                    allfolds_mse_test_per_subjectid = []
                    allfolds_train_nb_files_per_subjectid = []
                    allfolds_test_nb_files_per_subjectid = []

                    if bEveryoneSVR:
                        sFileName = '/objs_everyone_'+component+'_kernel_'+kernel+'_c_'+c_value+'_eps_'+epsilon+'.pkl'
                    else:
                        sFileName = '/objs_'+component+'_kernel_'+kernel+'_c_'+c_value+'_eps_'+epsilon+'.pkl'
                    print('sFileName : ', sFileName)

                    #To compute mean accuracy accross all folds
                    # To compute final score across all folds
                    allfolds_glob_trai_pred, allfolds_glob_trai_true, \
                    allfolds_glob_test_pred, \
                    allfolds_glob_test_true, \
                    allfolds_mse_training_per_subjectid, \
                    allfolds_mse_test_per_subjectid, \
                    allfolds_train_nb_files_per_subjectid, \
                    allfolds_test_nb_files_per_subjectid = get_all_folds(lResxFolders, sFileName)

                    global_training_r2 = r2_score(allfolds_glob_trai_true, allfolds_glob_trai_pred)
                    global_testing_r2 = r2_score(allfolds_glob_test_true, allfolds_glob_test_pred)
                    print('Global training R2: {}'.format(global_training_r2))
                    print('Global testing R2: {}'.format(global_testing_r2))
                    print('allfolds_mse_training_per_subjectid : ', len(allfolds_mse_training_per_subjectid))
                    print('SUM allfolds_mse_training_per_subjectid : ', sum(allfolds_train_nb_files_per_subjectid))
                    global_training_final_score = final_score(allfolds_mse_training_per_subjectid,
                                allfolds_train_nb_files_per_subjectid,
                                training_or_test='Train ')
                    global_testing_final_score = final_score(allfolds_mse_test_per_subjectid,
                                allfolds_test_nb_files_per_subjectid,
                                training_or_test='Test ')

                    if best_result.loc[0,'Test Final score'] > global_testing_final_score:
                        best_result.update({'Filename': [sFileName],
                                            'Global training r2':[global_training_r2],
                                            'Global testing r2':[global_testing_r2],
                                            'Train Final score':[global_training_final_score],
                                            'Test Final score':[global_testing_final_score]})
                
    
    print('------ GLOBAL WINNER PARAMETERS ------')
    print(best_result.transpose()[0].to_string())
    
def get_final_scores_SVR_lowest_mse_for_subjectid(sFilePath, bKnn, bSVR, bEveryoneSVR, bPerSubject, sDatabase, sSubchallenge):
    """
    Get final Scores for the individual SVR but we use different parameters for each patient depending on what 
    set of parameters worked best. 
    
    TODO: Not all arguments are needed, certain can be removed. 
    
    Keyword arguments:
    - TODO
    """
    lResxFolders, lObjsFiles = find_res_folders(sFilePath, bKnn, bSVR, bEveryoneSVR)
    lComponents, lKernel, lCValue, lEpsilon = find_components_neighbors(lObjsFiles, bKnn, bSVR, bEveryoneSVR)
    
    # QUICKFIX: Because right now we are not experimenting with different Kernels. Otherwise we need to add for loops 
    kernel=lKernel[0]
    epsilon=lEpsilon[0]
    
    # Each database and subchallenge have different subject_id 
    if sDatabase == "CIS" and sSubchallenge =="onoff":
        pids = np.array([1004,1006,1007,1019,1020,1023,1032,1034,1038,1039,1043,1044,1048,1049,1051])
    elif sDatabase == "CIS" and sSubchallenge =="tremor":
        pids = np.array([1004,1006,1007,1019,1020,1023,1032,1034,1038,1043,1046,1048,1049])
    elif sDatabase == "CIS" and sSubchallenge =="dysk":
        pids = np.array([1004,1007,1019,1023,1034,1038,1039,1043,1044,1048,1049])
        
    config_to_choose = {i:('FileName', 100) for i in pids}
    
    # First Step : Choose the best configuration for each patient 
    for subjectid_index in range(len(pids)):
        print('------ PATIENT ', pids[subjectid_index], ' ------')
        for c_value in lCValue:
            print('------ FOR C VALUE ', c_value, '------')
            for component in lComponents:
                print('---- FOR COMPONENT ', component, '----')

                if bEveryoneSVR:
                    sFileName = '/objs_everyone_'+component+'_kernel_'+kernel+'_c_'+c_value+'_eps_'+epsilon+'.pkl'
                else:
                    sFileName = '/objs_'+component+'_kernel_'+kernel+'_c_'+c_value+'_eps_'+epsilon+'.pkl'

                # Numpy array to contain the Test MSE for the 5 folds for one subject id 
                all_folds_mse_test_per_subjectid = []
                # List to contain the nb of test files for one subject id 
                total_nb_files = []

                for fold_folder in lResxFolders:
                    [glob_trai_pred,glob_trai_true, \
                     glob_test_pred,glob_test_true, \
                     mse_training_per_subjectid, \
                     mse_test_per_subjectid, \
                     train_nb_files_per_subjectid, \
                     test_nb_files_per_subjectid, \
                     vTestMeasurementId] = pickle.load(open(fold_folder+sFileName, "rb" ) )

                    # On Test Subset but we could do it on Train subset 
                    print('TEST MSE: ', mse_test_per_subjectid[subjectid_index])
                    total_nb_files.append(test_nb_files_per_subjectid[subjectid_index])
                    all_folds_mse_test_per_subjectid = np.append(all_folds_mse_test_per_subjectid, \
                                                                 mse_test_per_subjectid[subjectid_index])
                

                # Find the Challenge Final Score over the MSE per patient over the 5 folds
                # This candidate is just a way to get a metric to choose what is the best configuration 
                candidate_final_score = final_score(all_folds_mse_test_per_subjectid, total_nb_files, training_or_test='')
                print('Candidate Final Score : ', candidate_final_score) 
                if candidate_final_score < config_to_choose[pids[subjectid_index]][1]:
                    config_to_choose[pids[subjectid_index]] = [sFileName, candidate_final_score]
        
    print('------ GLOBAL WINNER PARAMETERS ------')
    pprint.pprint(config_to_choose)
    
    
    # Second Step: Pick that best configuration for all subjects id and output the global training final score
    # and the global testing final score 
    
    # To compute mean accuracy accross all folds
    allfolds_glob_trai_pred = []
    allfolds_glob_trai_true = []
    allfolds_glob_test_pred = []
    allfolds_glob_test_true = [] 

    # To compute final score across all folds 
    allfolds_mse_training_per_subjectid = []
    allfolds_mse_test_per_subjectid = []
    allfolds_train_nb_files_per_subjectid = []
    allfolds_test_nb_files_per_subjectid = []
    
    # FIXME to switch between training or data 
    lNb_Files_per_subjectid = test_nb_files_per_subjectid
    
    for fold_folder in lResxFolders:
        
        for subjectid_index in range(len(pids)):
            
            sFileName = config_to_choose[pids[subjectid_index]][0]
            
            [glob_trai_pred,glob_trai_true, \
             glob_test_pred,glob_test_true, \
             mse_training_per_subjectid, \
             mse_test_per_subjectid, \
             train_nb_files_per_subjectid, \
             test_nb_files_per_subjectid, \
             vTestMeasurementId] = pickle.load(open(fold_folder+sFileName, "rb" ) )

            allfolds_mse_training_per_subjectid = np.append(allfolds_mse_training_per_subjectid, 
                                                            mse_training_per_subjectid[subjectid_index])
            allfolds_mse_test_per_subjectid = np.append(allfolds_mse_test_per_subjectid,
                                                       mse_test_per_subjectid[subjectid_index])
            allfolds_train_nb_files_per_subjectid = np.append(allfolds_train_nb_files_per_subjectid, 
                                                             train_nb_files_per_subjectid[subjectid_index])
            allfolds_test_nb_files_per_subjectid = np.append(allfolds_test_nb_files_per_subjectid, 
                                                            test_nb_files_per_subjectid[subjectid_index])
    
    global_training_final_score = final_score(allfolds_mse_training_per_subjectid,
        allfolds_train_nb_files_per_subjectid,
        training_or_test='Train ')
    global_testing_final_score = final_score(allfolds_mse_test_per_subjectid,
            allfolds_test_nb_files_per_subjectid,
            training_or_test='Test ')          
                 
                    
                    
if __name__ == "__main__":
    # Usage example : 
    # sFilePath = '/home/sjoshi/codes/python/BeatPD/code/'
    # bKnn = True # Flag: is it files from KNN algorithm?  
    parser=argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars='@',
        description='Get Mean Accuracy and Final Score for the BeatPD Challenge.')

    parser.add_argument('--file-path',dest='sFilePath', required=True)
    parser.add_argument('--is-knn',dest='bKnn', default=False, required=False, action='store_true')
    parser.add_argument('--is-svr',dest='bSVR', default=False, required=False, action='store_true')
    parser.add_argument('--is-everyone-svr',dest='bEveryoneSVR', default=False, required=False, action='store_true')
    
    # Arguments for Per-Subject-SVR 
    parser.add_argument('--per-subject-svr',dest='bPerSubject', default=False, required=False, action='store_true')
    parser.add_argument('--database',dest='sDatabase', required=False)
    parser.add_argument('--subchallenge',dest='sSubchallenge', required=False)
    args=parser.parse_args()
    
    if args.bPerSubject:
        get_final_scores_SVR_lowest_mse_for_subjectid(**vars(args))
    elif args.bSVR or args.bEveryoneSVR:
        get_final_scores_SVR(**vars(args))
    else:
        print(args)
        get_final_scores_accuracy(**vars(args))
                                    
