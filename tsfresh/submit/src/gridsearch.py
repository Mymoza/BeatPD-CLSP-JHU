import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import sys
import xgboost as xgb
from xgboost import plot_importance
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor

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
# Drop the recordings where we don't have a label for the subchallenge
all_features_labels = all_features_labels.dropna(subset=[obj])

# Compute the average for a speaker of all the features and add those new features to all_features_label
avg = all_features_labels.groupby('subject_id').mean().reset_index().add_prefix('sp_').rename(columns={'sp_subject_id':'subject_id'})
all_features_labels = pd.merge(all_features_labels, avg, on='subject_id')

# Performing global normalization, removing the mean for the features and the labels
remove = []
for i in all_features_labels.columns:
    # if it doesn't start with sp and its sp_ is in  all_features_labels
    if not i.startswith('sp_') and 'sp_' + i in all_features_labels.columns:
        # We remove the sp_ column
        remove.append('sp_' + i)
        # We replace the feature with their value minus the mean of the subject 
        all_features_labels[i] = all_features_labels[i] - all_features_labels['sp_' + i]
# We drop the sp_ columns which were only useful to substract the mean
all_features_labels = all_features_labels.drop(remove, axis=1)

# The following comment was left there by Nanxin 
#al_y = all_features_labels[obj].astype(int)
#al = all_features_labels.drop([obj, 'measurement_id'], axis=1)
#mean_value = all_features_labels.groupby('subject_id').mean().reset_index().add_suffix('_mean')
#mean_value.columns = ['subject_id' if x=='subject_id_mean' else x for x in mean_value.columns]

# This adds the fold_id as a column 
all_features_labels = pd.merge(all_features_labels, pd.read_csv('data/order.csv'), how='inner', on=["measurement_id"])
# Count the number of recordings per subject_id per fold in a column called "spcount"
weight = all_features_labels.groupby(['subject_id', 'fold_id']).count().reset_index()[["subject_id", "fold_id", obj]].rename(columns={obj: 'spcount'})
# Add the number of recordings per subject to the all_features_label DataFrame
all_features_labels = pd.merge(all_features_labels, weight, on=['subject_id', 'fold_id'])
# Get a one hot encoding of which recording belongs to which subject_id
subject_id = pd.get_dummies(all_features_labels.subject_id, columns='subject_id', prefix='spk_')
##all_features_labels = pd.merge(all_features_labels, mean_value, on='subject_id')
# Concat the binary subject_id column to say which recording belongs to who 
all_features_labels = pd.concat([all_features_labels, subject_id], axis=1)
##all_features_labels = all_features_labels.astype(pd.np.float32)

# Y contains the labels for the obj (subchallenge) 
Y = np.array(all_features_labels[obj])
# FIXME: Why do we do the number of recordings times *-0.5? 
W = np.array(all_features_labels['spcount']) ** -0.5
foldid = np.array(all_features_labels['fold_id']).astype(int)
from sklearn.model_selection import PredefinedSplit
cv = PredefinedSplit(foldid)

# X contains all the features 
X = all_features_labels.drop([obj, 'subject_id', 'measurement_id', 'spcount', 'fold_id'], axis=1).astype(pd.np.float32).values
print(X.shape)
print(foldid.shape)

# Params Grid for xgboost
param_grid = {
        'objective': ['reg:squarederror'],
        'silent': [False],
        'max_depth': [2, 3, 4, 5, 6],
        'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2, 0.3],
        'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bylevel': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [0.5, 1.0, 3.0, 5.0, 7.0, 10.0],
        'gamma': [0, 0.25, 0.5, 1.0],
        'reg_lambda': [0.1, 1.0, 5.0, 10.0, 50.0, 100.0],
        'n_estimators': [50, 100, 500, 1000]}

# Params grid for a Random Forest Regressor
param_grid = {
    'n_estimators': [50, 55, 63, 64, 65, 67, 71, 74, 75, 78, 83, 86, 93, 95, 96, 97, 100],# 500, 1000],
    'max_depth': [5, 8, 15, 25, 30],
    'min_samples_split': [2, 5, 10, 15, 100],
    'min_samples_leaf': [1, 2, 5, 10],
    #'max_features': ['auto','0.25']
}

# FIXME: Uncomment this section to perform the gridsearch 
from sklearn.model_selection import RandomizedSearchCV
# xgboost
# clf = xgb.XGBRegressor()

# Random Forest Regressor 
clf = RandomForestRegressor()
rs_clf = RandomizedSearchCV(clf, param_grid, n_iter=100,
                            n_jobs=8, verbose=2, cv=cv,
                            refit=False, random_state=42, scoring='neg_mean_squared_error')
rs_clf.fit(X, Y, sample_weight=W) 
best_params = rs_clf.best_params_
print('Best Params : ')
print(best_params)

# Best parameters used for submission 3
# best_params = {'subsample': 1.0, 'silent': False, 'gamma': 1.0, 'reg_lambda': 100.0, 'min_child_weight': 0.5, 'objective': 'reg:squarederror', 'learning_rate': 0.3, 'max_depth': 2, 'colsample_bytree': 0.8, 'n_estimators': 100, 'colsample_bylevel': 0.5}

# For the random Forest regressor
# best_params = {'max_depth': 2, 'n_estimators': 100}


# with open('mdl/cis-pd.conf','wb') as f:
#     pickle.dump(best_params, f)

# Comment this exit if you want to get predictions files on test kfolds
#exit() 

########################################################################
# The following section is only used if you want to generate a predictions files
# on the test kfolds 
######################################################################

results = []
baselines = []

preds = []

lambda_value = None
#lambda_value = -3

for i in range(5):
    ##test = pd.read_csv(sys.argv[i]).squeeze()
    ##idx = all_features_labels['measurement_id'].isin(test)

    # Filter all_features_labels to only the measurements of the current fold
    idx = all_features_labels['fold_id'] == i

    ##tr_w = all_features_labels[~idx].groupby('subject_id').count().reset_index()[["subject_id", obj]].rename(columns={obj: 'spcount'})
    ##tr = pd.merge(all_features_labels[~idx], tr_w, on='subject_id')
    
    # Drop the measurements that are not from the current fold 
    tr = all_features_labels[~idx].drop(['fold_id'], axis=1)
    train_weight = tr['spcount'] ** -0.5 # training weight

    # Data augmentation with a lambda 
    if lambda_value is not None:
        # Take all the columns except the ones that are int and we don't want to multiply
        # FIXME: Do we want to multiple spcount?
        print('len tr keys : ', len(tr.columns))
        df_feat_mul = tr[tr.columns.difference(['subject_id','spcount'])]
        print('len df_feat_mul keys : ', len(df_feat_mul.columns))
        numerics = ['int16', 'int32', 'int64']

        newdf = df_feat_mul.select_dtypes(include=numerics)
        print(newdf.columns)
        # Multiply the features by the lambda_value 
        df_feat_mul = df_feat_mul.mul(lambda_value)
        # We add the subject_id, spcount columns back to the df with mutliplied values
        df_feat_mul = pd.concat([tr[['subject_id','spcount']], df_feat_mul], axis=1, sort=False)
        # Concatenate the original training features and the lambda multiplied ones 
        tr = pd.concat([tr, df_feat_mul])
        train_weight = tr['spcount'] ** -0.5 # training weight

    train_y = tr[obj].astype(pd.np.float32) # training labels 
    tr = tr.drop([obj, 'subject_id', 'measurement_id', 'spcount'], axis=1).astype(pd.np.float32)


    ##te_w = all_features_labels[idx].groupby('subject_id').count().reset_index()[["subject_id", obj]].rename(columns={obj: 'spcount'})
    ##te = pd.merge(all_features_labels[idx], te_w, on='subject_id')
    
    # Drop the measurements that are used in the training of this fold, so we keep [idx] instead of [~idx]
    te = all_features_labels[idx].drop(['fold_id'], axis=1)
    test_weight = te['spcount'] ** -0.5 # test weight 
    test_y = te[obj].astype(pd.np.float32) # testing labels
    #sub = te['subject_id']
    sid = te.subject_id
    test_measurement_id = te.measurement_id
    te = te.drop([obj, 'subject_id', 'measurement_id', 'spcount'], axis=1).astype(pd.np.float32)

    #clf = xgb.XGBRegressor(**best_params)
    #clf = xgb.XGBClassifier(**params)

    clf = RandomForestRegressor(**best_params)
    clf.fit(tr, train_y, sample_weight=train_weight)

    # Fit for the xgboost 
    # clf.fit(
    #     tr, train_y,
    #     sample_weight=train_weight,
    #     eval_set=[(tr, train_y), (te, test_y)],
    #     #eval_metric=',
    #     sample_weight_eval_set=[train_weight, test_weight],
    #     verbose=0,
    #     early_stopping_rounds=100
    # )
    pred = clf.predict(te).clip(0, 4)
    mse = (pred - test_y) ** 2
    #mse = test_y.to_numpy() ** 2
    mse2 = test_y ** 2
    #ret = pd.concat([sub, mse], axis=1)
    #ret.to_csv('tmp.csv')
    #mse = (ret.groupby('subject_id').mean())[obj].to_numpy()
    #cnt = (ret.groupby('subject_id').count())[obj].to_numpy()
    #cnt = cnt ** 0.5
    res = pd.DataFrame(data={'measurement_id': test_measurement_id, 'subject_id': sid, obj: pred})
    res = pd.merge(res, avg, on='subject_id')
    res[obj] += res['sp_' + obj]
    res = res[["measurement_id", obj]]
    preds.append(res)
    results.append(((mse * test_weight).sum() / test_weight.sum()).squeeze())
    baselines.append(((mse2 * test_weight).sum() / test_weight.sum()).squeeze())
preds = pd.concat(preds)
preds.to_csv('kfold_prediction_rf_cis-pd_{0}.csv'.format(obj), index=False)
#preds.to_csv('kfold_prediction_lambda_0.3_cis-pd_{0}.csv'.format(obj), index=False)
#print(clf.get_booster().get_score(importance_type='gain'))
print("baseline {0} result {1}".format(np.mean(baselines),np.mean(results)))
xgb.plot_importance(clf,max_num_features=10)
plt.savefig('importance_{0}.png'.format(obj))
