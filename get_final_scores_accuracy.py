#!/usr/bin/env python3.7
import pickle
import re 
import numpy as np
import glob
import argparse
from sklearn.metrics import mean_squared_error
from math import sqrt 
import pandas as pd 

def final_score(mse_per_subjectid, nb_files_per_subject_id, training_or_test=''):
    numerator = np.sum([nb_file * mse for nb_file, mse in zip(np.sqrt(nb_files_per_subject_id), mse_per_subjectid)])
    denominator = np.sum(np.sqrt(nb_files_per_subject_id))
    print(training_or_test+'Final score : ', np.divide(numerator, denominator))
    return np.divide(numerator, denominator)

def find_components_neighbors(lObjsFiles, bKnn, bSVR): 
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
        elif bSVR:
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
    elif bSVR:
        print('Kernels found : ', lKernel)
        print('C Values found : ', lCValue) 
        print('Epsilon found : ', lEpsilon) 

        return lComponents, lKernel, lCValue, lEpsilon
        
    return lNeighbors, lComponents
    
def find_res_folders(sFilePath, bKnn, bSVR):
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
    else:
        sFolderNamePattern = "resx*"
    
    print('Looking for folder : ', sFolderNamePattern)
    lResxFolders = [sFilePath+f for f in os.listdir(path) if re.search(sFolderNamePattern, f)]
    sPatternFold = '(?<=[Ff]old)\d+'

    # Get a list of all files starting with objs
    lObjsFiles = []
    for fold_folder in lResxFolders:
        if bKnn:
            lObjsFiles.extend(glob.glob(fold_folder + "/objs*k_*"))
        elif bSVR: 
            lObjsFiles.extend(glob.glob(fold_folder + "/objs*kernel_*"))

    print('lResxFolders : ', lResxFolders)
    print('lObjsFiles : ', lObjsFiles)
    return lResxFolders, lObjsFiles

def get_all_folds(lResxFolders, sFileName):
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
        
    
def get_final_scores_accuracy(sFilePath, bKnn, bSVR):
    """
    Read a pickle file and outputs the global mean accuracy & final score for BeatPD Challenge
    
    Keyword arguments:
    - sFilePath: Path to where the res* folders are. 
    - bKnn: Flag to say if the KNN algorithm is used to go through neighbors combination
    """
    
    lResxFolders, lObjsFiles = find_res_folders(sFilePath, bKnn, bSVR)
    lNeighbors, lComponents = find_components_neighbors(lObjsFiles, bKnn, bSVR)
    
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

def get_final_scores_SVR(sFilePath, bKnn, bSVR):
    """
    Read a pickle file and outputs the global mean accuracy & final score for BeatPD Challenge
    
    Keyword arguments:
    - sFilePath: Path to where the res* folders are. 
    - bKnn: Flag to say if the KNN algorithm is used to go through neighbors combination
    """
    
    lResxFolders, lObjsFiles = find_res_folders(sFilePath, bKnn, bSVR)
    lComponents, lKernel, lCValue, lEpsilon = find_components_neighbors(lObjsFiles, bKnn, bSVR)
    
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

                    global_training_accuracy = (allfolds_glob_trai_true == allfolds_glob_trai_pred).mean()
                    global_testing_accuracy = (allfolds_glob_test_true == allfolds_glob_test_pred).mean()
                    print('Global training R2: {}'.format(global_training_accuracy))
                    print('Global testing R2: {}'.format(global_testing_accuracy))

                    global_training_final_score = final_score(allfolds_mse_training_per_subjectid,
                                allfolds_train_nb_files_per_subjectid,
                                training_or_test='Train ')
                    global_testing_final_score = final_score(allfolds_mse_test_per_subjectid,
                                allfolds_test_nb_files_per_subjectid,
                                training_or_test='Test ')

                    if best_result.loc[0,'Test Final score'] > global_testing_final_score:
                        best_result.update({'Filename': [sFileName],
                                            'Global training r2':[global_training_accuracy],
                                            'Global testing r2':[global_testing_accuracy],
                                            'Train Final score':[global_training_final_score],
                                            'Test Final score':[global_testing_final_score]})
                
    
    print('------ GLOBAL WINNER PARAMETERS ------')
    print(best_result.transpose()[0].to_string())
    
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
    args=parser.parse_args()
    
    if args.bSVR:
        print('yayyyyyyyyy good if') 
        get_final_scores_SVR(**vars(args))
    else:
        print('KNN SIDE YAY')
        print(args)
        get_final_scores_accuracy(**vars(args))
                                    
