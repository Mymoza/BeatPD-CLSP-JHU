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

# contains the subchallenge we are working on 
obj = sys.argv[1]

# All the subchallenges
all_obj = ["on_off", "tremor", "dyskinesia"]

# Read the training features 
all_features = pd.read_csv(sys.argv[2])

# Read the training labels
all_labels = pd.read_csv(sys.argv[3])
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
all_features_labels = pd.merge(all_features_labels, pd.read_csv('data/order.csv'), how='inner', on=["measurement_id"])
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

#FIXME: The following code is not complete though. 
# It should loop over the subject_ids and read all of their best_params,
# make predictions and append those to a variable which we put to a csv at the end 

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
        # FIXME: is there really a prediction column? Should i instead drop obj?
        test_spk = test_spk.drop(['measurement_id', 'subject_id', 'prediction', 'spcount', 'fold_id'], axis=1).astype(pd.np.float32)

        # Using the saved configuration provided at the top of this document
        clf = xgb.XGBRegressor(**cfgs[spk_id])
        
        # Currently Using Stop Criteria on Training Data for the speaker
        # FIXME: Why are we not using (te, test_y) in eval_set ?
        clf.fit(
            train_spk, train_y,
            sample_weight=train_weight,
            eval_set=[(train_spk, train_y)],
            #eval_metric=',
            sample_weight_eval_set=[train_weight],
            verbose=0,
            early_stopping_rounds=100
        )
        # Append the predictions for every speaker for each fold
        results = pd.DataFrame(data={obj: clf.predict(test_spk).clip(0,4)})
        # Join test_measurement_id, results and test_subject_id
        joined = pd.DataFrame(test_measurement_id).join(results).join(test_subject_id)
        # FIXME: why we add avg to the preds?
        joined = pd.merge(joined, avg, on='subject_id')
        joined[obj] += joined['sp_' + obj]
        joined = joined[['measurement_id', obj]]
        A.append(joined)
joined = pd.concat(A)
joined.to_csv('kfold_prediction_cis-pd.{0}.perpatient.csv'.format(obj), index=False)
