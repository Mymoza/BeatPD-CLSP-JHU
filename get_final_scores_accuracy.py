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

def find_components_neighbors(sFilePath, bKnn):
    # Building the list of folders we have to open 
    sFolderName = ("resi*" if bKnn else "resx*")
    print('Looking for folder : ', sFolderName)
    lResxFolders = [f for f in glob.glob(sFilePath + sFolderName)]
    sPatternFold = '(?<=[Ff]old)\d+'
    lComponents = [] 
    lNeighbors = [] 

    # Get a list of all files starting with objs
    for fold_folder in lResxFolders:
        lObjsFiles = glob.glob(fold_folder + "/objs*")

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
            noToAdd = re.findall(sPatternNeighbors, objsFile)[0]
            lNeighbors.append(noToAdd) if noToAdd not in lNeighbors else lNeighbors
        else:
            # Quickfix to ignore the Neighbors loop if an algorithm other than KNN is used 
            lNeighbors = [1]
    
    print('Components found : ', lComponents)
    (print('Neighbors found : ', lNeighbors) if bKnn else '')
    
    return lNeighbors, lComponents, lResxFolders

def get_final_scores_accuracy(sFilePath, bKnn):
    """
    Read a pickle file and outputs the global mean accuracy & final score for BeatPD Challenge
    
    Keyword arguments:
    - sFilePath: Path to where the resx* folders are. 
    - bKnn: Flag to say if the KNN algorithm is used to go through neighbors combination
    """
    
    lNeighbors, lComponents, lResxFolders = find_components_neighbors(sFilePath, bKnn)
    
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
            
            for fold_folder in lResxFolders:
                print(fold_folder)

                if bKnn:
                    sFileName = '/objs_'+component+'_k_'+neighbor+'.pkl'
                else:
                    sFileName = '/objs_'+component+'.pkl'

                # Retrieve DataFrames from Pickle file 
                [glob_trai_pred,glob_trai_true, \
                 glob_test_pred,glob_test_true, \
                 mse_training_per_subjectid, \
                 mse_test_per_subjectid, \
                 train_nb_files_per_subjectid, \
                 test_nb_files_per_subjectid] = pickle.load(open(fold_folder+sFileName, "rb" ) )
                 # vTestMeasurementId] = pickle.load(open(fold_folder+sFileName, "rb" ) )

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
    
    
    # Loading the file with that obtained the best results 
    # iNeighbor = best_result 
    # iComponents = best_result 
    for neighbor in lNeighbors:
        (print('------ FOR NEIGHBOR ', neighbor, '------') if bKnn else '')
        for component in lComponents:
            print('---- FOR COMPONENT ', component, '----')
            
            for fold_folder in lResxFolders:
            
                [glob_trai_pred,glob_trai_true, \
                             glob_test_pred,glob_test_true, \
                             mse_training_per_subjectid, \
                             mse_test_per_subjectid, \
                             train_nb_files_per_subjectid, \
                             test_nb_files_per_subjectid] = pickle.load(open(fold_folder+best_result.Filename[0], "rb" ) )
                             #vTestMeasurementId] = pickle.load(open(fold_folder+best_result.Filename[0], "rb" ) )

             # Build the DataFrames for all folds 
                allfolds_glob_trai_pred = np.append(allfolds_glob_trai_pred, glob_trai_pred)
                allfolds_glob_trai_true = np.append(allfolds_glob_trai_true, glob_trai_true)
                allfolds_glob_test_pred = np.append(allfolds_glob_test_pred, glob_test_pred)
#                 allfolds_glob_test_true = np.append(allfolds_glob_test_true, glob_test_true)

                allfolds_mse_training_per_subjectid = np.append(allfolds_mse_training_per_subjectid, 
                                                                mse_training_per_subjectid)
#                 allfolds_mse_test_per_subjectid = np.append(allfolds_mse_test_per_subjectid,
#                                                            mse_test_per_subjectid)
                allfolds_train_nb_files_per_subjectid = np.append(allfolds_train_nb_files_per_subjectid, 
                                                                 train_nb_files_per_subjectid)
                allfolds_test_nb_files_per_subjectid = np.append(allfolds_test_nb_files_per_subjectid, 
                                                                test_nb_files_per_subjectid)
    
    # Concatenating the measurement_id and the predictions 
    pdTestPredictions = pd.concat([pd.DataFrame(vTestMeasurementId),
                                   pd.DataFrame(allfolds_glob_test_pred)], axis=1)
    
    pdTestPredictions.to_csv(
        sOut + 'predictions_'+str(iComponents)+'_k_'+str(iNeighbors)+".csv",
        index=False,
        header=["measurement_id","prediction"],
    )

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

    args=parser.parse_args()
    
    get_final_scores_accuracy(**vars(args))
                                    
