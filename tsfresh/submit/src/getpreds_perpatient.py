####################################
#
####################################

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import sys
import xgboost as xgb
from xgboost import plot_importance
import numpy as np
import pickle
import datetime
import argparse

parser = argparse.ArgumentParser(description='Perform gridsearch or predicts on test folds.')
parser.add_argument('symptom', metavar='obj', type=str, help='Should be either on_off, tremor, or dyskinesia')
parser.add_argument("--features", action="append", type=str, help='Path to the features, like features/cis-pd.training.csv')
parser.add_argument("--labels", type=str, help='Path to the labels, for example \{\path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv')
parser.add_argument("--filename", type=str, help='filename')
parser.add_argument("--pred_path", type=str, help='path to pred files')
args = parser.parse_args()

# contains the subchallenge we are working on 
obj = args.symptom

# All the subchallenges
all_obj = ["on_off", "tremor", "dyskinesia"]

# Read the training features 
print(args.features)
all_features = pd.concat((pd.read_csv(f) for f in args.features))

# Read the training labels
all_labels = pd.read_csv(args.labels)
# Drop the labels of the subchallenges we're not working on 
all_labels = all_labels.drop(list(set(all_obj) - set([obj])), axis=1)

# Merge the features and the labels on measurement_id
all_features_labels = pd.merge(all_features, all_labels, on=["measurement_id"])
all_features_labels = all_features_labels.dropna(subset=[obj])
avg = all_features_labels.groupby('subject_id').mean().reset_index().add_prefix('sp_').rename(columns={'sp_subject_id':'subject_id'})
all_features_labels = pd.merge(all_features_labels, avg, on='subject_id')
remove = []
for i in all_features_labels.columns:
    if not i.startswith('sp_') and 'sp_' + i in all_features_labels.columns:
        remove.append('sp_' + i)
        all_features_labels[i] = all_features_labels[i] - all_features_labels['sp_' + i]
all_features_labels = all_features_labels.drop(remove, axis=1)
#al_y = all_features_labels[obj].astype(int)
#all_features_labels = all_features_labels.drop([obj, 'measurement_id'], axis=1)
#mean_value = all_features_labels.groupby('subject_id').mean().reset_index().add_suffix('_mean')
#mean_value.columns = ['subject_id' if x=='subject_id_mean' else x for x in mean_value.columns]
all_features_labels = pd.merge(all_features_labels, pd.read_csv('/home/mpgill/BeatPD/BeatPD-CLSP-JHU/tsfresh/submit/data/order.csv'), how='inner', on=["measurement_id"])
weight = all_features_labels.groupby(['subject_id', 'fold_id']).count().reset_index()[["subject_id", "fold_id", obj]].rename(columns={obj: 'spcount'})
all_features_labels = pd.merge(all_features_labels, weight, on=['subject_id', 'fold_id'])
#subject_id = pd.get_dummies(all_features_labels.subject_id, columns='subject_id', prefix='spk_')
#all_features_labels = pd.merge(all_features_labels, mean_value, on='subject_id')
#all_features_labels = pd.concat([all_features_labels, subject_id], axis=1)
#all_features_labels = all_features_labels.astype(pd.np.float32)
all_spks = all_features_labels['subject_id'].unique()


# The following section hardcode the best hyperparameters we used for the 4th submission with per patient tuning 
cfgs = {}
#tremor
if obj == 'tremor':
    cfgs[1004] = {'colsample_bytree': 0.8, 'reg_lambda': 10.0, 'colsample_bylevel': 0.7, 'subsample': 0.5, 'max_depth': 4, 'learning_rate': 0.2, 'n_estimators': 100, 'objective': 'reg:squarederror', 'silent': False, 'min_child_weight': 1.0, 'gamma': 0.5}
    cfgs[1006] = {'silent': False, 'subsample': 0.5, 'gamma': 0.25, 'min_child_weight': 5.0, 'reg_lambda': 0.1, 'objective': 'reg:squarederror', 'max_depth': 3, 'n_estimators': 100, 'colsample_bytree': 0.8, 'colsample_bylevel': 0.8, 'learning_rate': 0.3}
    cfgs[1007] = {'max_depth': 5, 'colsample_bytree': 0.8, 'learning_rate': 0.1, 'min_child_weight': 5.0, 'n_estimators': 500, 'silent': False, 'subsample': 0.8, 'colsample_bylevel': 0.8, 'gamma': 0.5, 'reg_lambda': 0.1, 'objective': 'reg:squarederror'}
    cfgs[1019] = {'reg_lambda': 10.0, 'subsample': 0.8, 'max_depth': 3, 'gamma': 0, 'colsample_bylevel': 0.6, 'n_estimators': 100, 'silent': False, 'min_child_weight': 3.0, 'colsample_bytree': 0.4, 'learning_rate': 0.1, 'objective': 'reg:squarederror'}
    cfgs[1020] = {'reg_lambda': 0.1, 'colsample_bytree': 0.8, 'subsample': 0.7, 'gamma': 0, 'colsample_bylevel': 0.9, 'max_depth': 2, 'learning_rate': 0.1, 'objective': 'reg:squarederror', 'n_estimators': 100, 'silent': False, 'min_child_weight': 0.5}
    cfgs[1023] = {'subsample': 0.9, 'colsample_bylevel': 0.5, 'min_child_weight': 10.0, 'gamma': 0.5, 'n_estimators': 50, 'learning_rate': 0.3, 'silent': False, 'reg_lambda': 5.0, 'colsample_bytree': 1.0, 'max_depth': 6, 'objective': 'reg:squarederror'}
    cfgs[1032] = {'gamma': 0.25, 'subsample': 1.0, 'objective': 'reg:squarederror', 'silent': False, 'colsample_bytree': 1.0, 'learning_rate': 0.3, 'max_depth': 2, 'reg_lambda': 5.0, 'colsample_bylevel': 0.5, 'min_child_weight': 7.0, 'n_estimators': 500}
    cfgs[1034] = {'silent': False, 'min_child_weight': 5.0, 'learning_rate': 0.2, 'colsample_bylevel': 0.5, 'max_depth': 3, 'objective': 'reg:squarederror', 'colsample_bytree': 0.9, 'gamma': 0, 'subsample': 0.9, 'n_estimators': 50, 'reg_lambda': 5.0}
    cfgs[1038] = {'objective': 'reg:squarederror', 'colsample_bylevel': 0.8, 'learning_rate': 0.2, 'subsample': 0.8, 'silent': False, 'min_child_weight': 7.0, 'gamma': 0.5, 'max_depth': 6, 'reg_lambda': 5.0, 'colsample_bytree': 0.9, 'n_estimators': 500}
    cfgs[1043] = {'subsample': 0.5, 'objective': 'reg:squarederror', 'colsample_bylevel': 0.8, 'min_child_weight': 3.0, 'reg_lambda': 5.0, 'silent': False, 'n_estimators': 1000, 'max_depth': 6, 'colsample_bytree': 0.8, 'gamma': 1.0, 'learning_rate': 0.2}
    cfgs[1046] = {'colsample_bylevel': 0.4, 'subsample': 0.5, 'min_child_weight': 5.0, 'n_estimators': 1000, 'colsample_bytree': 0.6, 'reg_lambda': 0.1, 'max_depth': 6, 'learning_rate': 0.05, 'objective': 'reg:squarederror', 'silent': False, 'gamma': 0.5}
    cfgs[1048] = {'silent': False, 'objective': 'reg:squarederror', 'gamma': 0.5, 'learning_rate': 0.3, 'colsample_bylevel': 0.5, 'n_estimators': 50, 'subsample': 0.6, 'colsample_bytree': 0.6, 'min_child_weight': 10.0, 'reg_lambda': 5.0, 'max_depth': 2}
    cfgs[1049] = {'silent': False, 'n_estimators': 100, 'learning_rate': 0.2, 'objective': 'reg:squarederror', 'reg_lambda': 10.0, 'colsample_bytree': 0.8, 'colsample_bylevel': 0.7, 'gamma': 0.5, 'max_depth': 4, 'min_child_weight': 1.0, 'subsample': 0.5}
if obj == 'dyskinesia':
    cfgs[1004] = {'objective': 'reg:squarederror', 'colsample_bytree': 0.8, 'reg_lambda': 10.0, 'min_child_weight': 1.0, 'colsample_bylevel': 0.7, 'gamma': 0.5, 'learning_rate': 0.2, 'silent': False, 'subsample': 0.5, 'max_depth': 4, 'n_estimators': 100}
    cfgs[1007] = {'min_child_weight': 10.0, 'objective': 'reg:squarederror', 'gamma': 0, 'max_depth': 4, 'reg_lambda': 10.0, 'colsample_bytree': 0.6, 'colsample_bylevel': 0.7, 'learning_rate': 0.1, 'n_estimators': 50, 'silent': False, 'subsample': 0.9}
    cfgs[1019] = {'silent': False, 'max_depth': 3, 'min_child_weight': 1.0, 'subsample': 0.9, 'colsample_bytree': 0.9, 'reg_lambda': 10.0, 'learning_rate': 0.05, 'n_estimators': 1000, 'objective': 'reg:squarederror', 'gamma': 0, 'colsample_bylevel': 0.4}
    cfgs[1023] = {'colsample_bylevel': 0.9, 'reg_lambda': 0.1, 'min_child_weight': 0.5, 'n_estimators': 100, 'subsample': 0.7, 'colsample_bytree': 0.8, 'silent': False, 'max_depth': 2, 'learning_rate': 0.1, 'objective': 'reg:squarederror', 'gamma': 0}
    cfgs[1034] = {'learning_rate': 0.3, 'min_child_weight': 1.0, 'subsample': 0.8, 'objective': 'reg:squarederror', 'silent': False, 'n_estimators': 1000, 'max_depth': 5, 'colsample_bylevel': 0.7, 'gamma': 0, 'reg_lambda': 1.0, 'colsample_bytree': 0.5}
    cfgs[1038] = {'learning_rate': 0.2, 'subsample': 0.8, 'n_estimators': 50, 'reg_lambda': 5.0, 'colsample_bylevel': 1.0, 'silent': False, 'min_child_weight': 7.0, 'colsample_bytree': 0.8, 'max_depth': 6, 'objective': 'reg:squarederror', 'gamma': 0.5}
    cfgs[1039] = {'colsample_bytree': 0.6, 'max_depth': 6, 'learning_rate': 0.05, 'silent': False, 'min_child_weight': 5.0, 'colsample_bylevel': 0.4, 'reg_lambda': 0.1, 'subsample': 0.5, 'objective': 'reg:squarederror', 'gamma': 0.5, 'n_estimators': 1000}
    cfgs[1043] = {'silent': False, 'reg_lambda': 0.1, 'max_depth': 2, 'learning_rate': 0.05, 'subsample': 1.0, 'min_child_weight': 5.0, 'gamma': 0, 'n_estimators': 500, 'colsample_bylevel': 0.9, 'colsample_bytree': 1.0, 'objective': 'reg:squarederror'}
    cfgs[1044] = {'colsample_bylevel': 0.4, 'learning_rate': 0.3, 'silent': False, 'objective': 'reg:squarederror', 'min_child_weight': 10.0, 'subsample': 0.8, 'max_depth': 3, 'colsample_bytree': 0.4, 'n_estimators': 100, 'gamma': 0, 'reg_lambda': 50.0}
    cfgs[1048] = {'colsample_bylevel': 0.7, 'learning_rate': 0.3, 'min_child_weight': 5.0, 'reg_lambda': 0.1, 'subsample': 1.0, 'max_depth': 2, 'colsample_bytree': 0.9, 'objective': 'reg:squarederror', 'silent': False, 'n_estimators': 100, 'gamma': 1.0}
    cfgs[1049] = {'colsample_bylevel': 0.5, 'n_estimators': 50, 'reg_lambda': 5.0, 'colsample_bytree': 0.9, 'gamma': 0, 'min_child_weight': 5.0, 'silent': False, 'objective': 'reg:squarederror', 'subsample': 0.9, 'learning_rate': 0.2, 'max_depth': 3}
if obj == 'on_off':
    cfgs[1004] = {'learning_rate': 0.3, 'colsample_bytree': 1.0, 'silent': False, 'gamma': 0.25, 'min_child_weight': 7.0, 'colsample_bylevel': 0.5, 'subsample': 1.0, 'objective': 'reg:squarederror', 'reg_lambda': 5.0, 'max_depth': 2, 'n_estimators': 500}
    cfgs[1006] = {'n_estimators': 500, 'colsample_bylevel': 1.0, 'gamma': 0, 'objective': 'reg:squarederror', 'reg_lambda': 5.0, 'max_depth': 6, 'subsample': 0.8, 'min_child_weight': 5.0, 'learning_rate': 0.2, 'silent': False, 'colsample_bytree': 1.0}
    cfgs[1007] = {'colsample_bytree': 0.4, 'max_depth': 3, 'subsample': 0.5, 'n_estimators': 50, 'learning_rate': 0.3, 'silent': False, 'min_child_weight': 7.0, 'objective': 'reg:squarederror', 'gamma': 0.25, 'reg_lambda': 50.0, 'colsample_bylevel': 0.4}
    cfgs[1019] = {'objective': 'reg:squarederror', 'max_depth': 3, 'colsample_bylevel': 0.4, 'n_estimators': 1000, 'gamma': 0, 'silent': False, 'min_child_weight': 1.0, 'subsample': 0.9, 'colsample_bytree': 0.9, 'reg_lambda': 10.0, 'learning_rate': 0.05}
    cfgs[1020] = {'learning_rate': 0.1, 'silent': False, 'colsample_bylevel': 0.7, 'subsample': 0.9, 'objective': 'reg:squarederror', 'reg_lambda': 10.0, 'min_child_weight': 10.0, 'gamma': 0, 'colsample_bytree': 0.6, 'n_estimators': 50, 'max_depth': 4}
    cfgs[1023] = {'colsample_bytree': 0.4, 'learning_rate': 0.1, 'objective': 'reg:squarederror', 'colsample_bylevel': 0.4, 'silent': False, 'subsample': 0.5, 'n_estimators': 100, 'reg_lambda': 0.1, 'gamma': 0.5, 'min_child_weight': 10.0, 'max_depth': 5}
    cfgs[1032] = {'reg_lambda': 1.0, 'colsample_bytree': 0.5, 'n_estimators': 1000, 'objective': 'reg:squarederror', 'colsample_bylevel': 1.0, 'max_depth': 3, 'silent': False, 'gamma': 0, 'learning_rate': 0.1, 'subsample': 1.0, 'min_child_weight': 3.0}
    cfgs[1034] = {'min_child_weight': 0.5, 'gamma': 0, 'learning_rate': 0.3, 'max_depth': 2, 'silent': False, 'reg_lambda': 100.0, 'colsample_bytree': 0.8, 'colsample_bylevel': 0.7, 'subsample': 0.7, 'n_estimators': 1000, 'objective': 'reg:squarederror'}
    cfgs[1038] = {'gamma': 0.25, 'colsample_bytree': 0.9, 'subsample': 0.6, 'colsample_bylevel': 0.4, 'objective': 'reg:squarederror', 'max_depth': 5, 'n_estimators': 100, 'min_child_weight': 5.0, 'reg_lambda': 50.0, 'learning_rate': 0.2, 'silent': False}
    cfgs[1039] = {'objective': 'reg:squarederror', 'min_child_weight': 7.0, 'n_estimators': 500, 'subsample': 1.0, 'silent': False, 'learning_rate': 0.3, 'colsample_bylevel': 0.5, 'max_depth': 2, 'reg_lambda': 5.0, 'gamma': 0.25, 'colsample_bytree': 1.0}
    cfgs[1043] = {'colsample_bytree': 0.8, 'gamma': 1.0, 'reg_lambda': 5.0, 'colsample_bylevel': 0.8, 'min_child_weight': 3.0, 'learning_rate': 0.2, 'subsample': 0.5, 'objective': 'reg:squarederror', 'max_depth': 6, 'n_estimators': 1000, 'silent': False}
    cfgs[1044] = {'max_depth': 6, 'subsample': 0.7, 'colsample_bylevel': 0.7, 'min_child_weight': 7.0, 'learning_rate': 0.05, 'reg_lambda': 1.0, 'n_estimators': 1000, 'colsample_bytree': 0.8, 'gamma': 0.25, 'silent': False, 'objective': 'reg:squarederror'}
    cfgs[1048] = {'colsample_bylevel': 0.9, 'colsample_bytree': 1.0, 'reg_lambda': 0.1, 'silent': False, 'learning_rate': 0.05, 'gamma': 0, 'objective': 'reg:squarederror', 'max_depth': 2, 'min_child_weight': 5.0, 'n_estimators': 500, 'subsample': 1.0}
    cfgs[1049] = {'colsample_bylevel': 1.0, 'learning_rate': 0.01, 'gamma': 0, 'max_depth': 2, 'objective': 'reg:squarederror', 'colsample_bytree': 1.0, 'reg_lambda': 0.1, 'n_estimators': 1000, 'min_child_weight': 5.0, 'subsample': 1.0, 'silent': False}
    cfgs[1051] = {'gamma': 1.0, 'colsample_bytree': 0.8, 'n_estimators': 1000, 'learning_rate': 0.2, 'objective': 'reg:squarederror', 'min_child_weight': 3.0, 'silent': False, 'max_depth': 6, 'subsample': 0.5, 'reg_lambda': 5.0, 'colsample_bylevel': 0.8}


####################################################
# The following section is only used to create a predictions files for 
# cis-pd with per patient tuning 
####################################################
results_mse = []
results = []
preds = []
A = []
for i in range(5):
    for spk in all_spks:
        spk_id = int(spk)

        # Filter all_features_labels to only the measurements of the current fold
        idx = all_features_labels['fold_id'] == i

        # Only keep the measurements of the speaker we're going to predict 
        train_spk = all_features_labels.loc[all_features_labels['subject_id'] == spk]
        # Drop the measurements that are not from the current fold 
        train_spk = train_spk[~idx]
        train_y = train_spk[obj].astype(pd.np.float32)
        train_weight = train_spk['spcount'] ** -0.5
        train_spk = train_spk.drop([obj, 'subject_id', 'measurement_id', 'spcount', 'fold_id'], axis=1).astype(pd.np.float32)

        test_spk = all_features_labels.loc[all_features_labels['subject_id'] == spk]
        test_spk = test_spk[idx]
        test_weight = test_spk['spcount'] ** -0.5 # test weight 
        test_measurement_id = test_spk.measurement_id.reset_index(drop=True)
        test_subject_id = test_spk.subject_id.reset_index(drop=True)
        test_spk_y = test_spk[obj].astype(pd.np.float32)
        test_spk = test_spk.drop(['measurement_id', 'subject_id', obj, 'spcount', 'fold_id'], axis=1).astype(pd.np.float32)
        # Using the saved configuration provided at the top of this document
        clf = xgb.XGBRegressor(**cfgs[spk_id])
        
        # Split the test dataset in two to make a development dataset 
        # print(len(test_spk))
        # print('-----------')
        # print(test_spk)
        # print('------------------')
        # print('initial : ', test_measurement_id)
        # test_measurement_id_yay = np.split(test_measurement_id, [int(.5*len(test_measurement_id))])
        # print('test_measurement_id_yay 0 : ', test_measurement_id_yay[0])
        # print('test_measurement_id_yay 1 : ', test_measurement_id_yay[1])
        # print('test_measurement_id_yay : ', test_measurement_id_yay)
        # print('type(test_measurement_id_yay) : ', type(test_measurement_id_yay))

        test_spk_dev = np.split(test_spk, [int(.5*len(test_spk))])
        test_spk_y_dev = np.split(test_spk_y, [int(.5*len(test_spk_y))])
        test_weight_dev = np.split(test_weight, [int(.5*len(test_weight))])
        # print('test_spk_dev ', test_spk_dev)
        # print('len test_spk_dev[0] ', len(test_spk_dev[0]))
        # print('len test_spk_dev[1] ', len(test_spk_dev[1]))
        # print('len test_spk_y_dev[0] : ', len(test_spk_y_dev[0]))
        # print('len test_spk_y_dev[1] : ', len(test_spk_y_dev[1]))
        # print("===================")
        # print('test_spk_dev[0] ', test_spk_dev[0])
        # print('test_spk_y_dev[0] : ', test_spk_y_dev[0])

        print('dev : ', len(test_spk_dev[0]),', test : ',len(test_spk_dev[1]),' = ', str(len(test_spk_dev[0])+len(test_spk_dev[1])))
        pred_test_fold = [] 
        # for index in range(2):
        for index in [0, 1]:
            # Currently Using Stop Criteria on Training Data for the speaker
            clf.fit(
                train_spk, train_y,
                sample_weight=train_weight,
                eval_set=[(train_spk, train_y), (test_spk_dev[index], test_spk_y_dev[index])], #[(train_spk, train_y)],
                #eval_metric=',
                sample_weight_eval_set=[train_weight, test_weight_dev[index]],
                verbose=0,
                early_stopping_rounds=100
            )
            # Predict on the dataset that was not used for the early stop
            pred = clf.predict(test_spk_dev[1 if index == 0 else 0]).clip(0,4)
            pred_test_fold.append(pred)

            # Trying to get Nanxin's MSE  to see if it's lower than my results
            mse = (pred - test_spk_y_dev[1 if index == 0 else 0]) ** 2
            results_mse.append(((mse * test_weight_dev[1 if index == 0 else 0]).sum() / test_weight_dev[1 if index == 0 else 0].sum()).squeeze())

        # print('len(test_measurement_id) : ', len(test_measurement_id))
        # print('len(pred_test_fold) ', len(pred_test_fold))
        # print('type(pred_test_fold[0]) : ', type(pred_test_fold[0]))
        # print('before : ', pred_test_fold)
        pred_test_fold = np.concatenate(pred_test_fold, axis=None)
        # print('after concatenate : ', pred_test_fold)
        # Append the predictions for every speaker for each fold
        results = pd.DataFrame(data={obj: np.array(pred_test_fold)}) # used to be pred 
        # Join test_measurement_id, results and test_subject_id
        joined = pd.DataFrame(test_measurement_id).join(results).join(test_subject_id)
        joined = pd.merge(joined, avg, on='subject_id')
        joined[obj] += joined['sp_' + obj] # we add the average back 
        joined = joined[['measurement_id', obj]]
        A.append(joined)
joined = pd.concat(A)
joined.to_csv(args.pred_path+'kfold_prediction_cis-pd.{0}_{1}.perpatient.csv'.format(obj, args.filename), index=False)

print(type(results_mse))
print(len(results_mse))
results_mse = pd.DataFrame(results_mse)
print(type(results_mse))
print(len(results_mse))
print("baseline result {0}".format(np.mean(results_mse)))
