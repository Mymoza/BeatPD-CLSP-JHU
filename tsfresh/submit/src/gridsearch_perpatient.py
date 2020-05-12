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

obj = sys.argv[1]
all_obj = ["on_off", "tremor", "dyskinesia"]

all_fea = pd.read_csv(sys.argv[2])
all_lab = pd.read_csv(sys.argv[3])
all_lab = all_lab.drop(list(set(all_obj) - set([obj])), axis=1)

al = pd.merge(all_fea, all_lab, on=["measurement_id"])
al = al.dropna(subset=[obj])
avg = al.groupby('subject_id').mean().reset_index().add_prefix('sp_').rename(columns={'sp_subject_id':'subject_id'})
al = pd.merge(al, avg, on='subject_id')
remove = []
for i in al.columns:
    if not i.startswith('sp_') and 'sp_' + i in al.columns:
        remove.append('sp_' + i)
        al[i] = al[i] - al['sp_' + i]
al = al.drop(remove, axis=1)
#al_y = al[obj].astype(int)
#al = al.drop([obj, 'measurement_id'], axis=1)
#mean_value = al.groupby('subject_id').mean().reset_index().add_suffix('_mean')
#mean_value.columns = ['subject_id' if x=='subject_id_mean' else x for x in mean_value.columns]
al = pd.merge(al, pd.read_csv('data/order.csv'), how='inner', on=["measurement_id"])
weight = al.groupby(['subject_id', 'fold_id']).count().reset_index()[["subject_id", "fold_id", obj]].rename(columns={obj: 'spcount'})
al = pd.merge(al, weight, on=['subject_id', 'fold_id'])
#subject_id = pd.get_dummies(al.subject_id, columns='subject_id', prefix='spk_')
#al = pd.merge(al, mean_value, on='subject_id')
#al = pd.concat([al, subject_id], axis=1)
#al = al.astype(pd.np.float32)
spks = al['subject_id'].unique()

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

ret = {}
for i in spks:
    print("{0}:{1}".format(datetime.datetime.now(), i))
    al_spk = al.loc[al['subject_id'] == int(i)]
    Y = al_spk[obj].to_numpy()
    W = al_spk['spcount'].to_numpy() ** -0.5
    foldid = al_spk['fold_id'].to_numpy().astype(int)
    from sklearn.model_selection import PredefinedSplit
    cv = PredefinedSplit(foldid)
    
    X = al_spk.drop([obj, 'measurement_id', 'spcount', 'fold_id'], axis=1).astype(pd.np.float32).to_numpy()
    
    
    from sklearn.model_selection import RandomizedSearchCV
    clf = xgb.XGBRegressor()
    rs_clf = RandomizedSearchCV(clf, param_grid, n_iter=100,
                                n_jobs=8, verbose=1, cv=cv,
                                refit=False, random_state=42, scoring='neg_mean_squared_error')
    rs_clf.fit(X, Y, sample_weight=W) 
    best_params = rs_clf.best_params_
    print(i)
    print(best_params)
    #print(rs_clf.beat_score_)
    with open('mdl/cis-pd.'+obj+'.'+str(i)+'.conf', 'wb') as f:
        pickle.dump(best_params, f)

#best_params = {'min_child_weight': 3.0, 'learning_rate': 0.1, 'n_estimators': 50, 'colsample_bylevel': 0.5, 'objective': 'reg:squarederror', 'subsample': 0.6, 'max_depth': 3, 'gamma': 0.5, 'silent': False, 'colsample_bytree': 0.4, 'reg_lambda': 100.0}

#best_params = {'min_child_weight': 5.0, 'objective': 'reg:squarederror', 'colsample_bylevel': 0.4, 'subsample': 1.0, 'gamma': 1.0, 'silent': False, 'reg_lambda': 100.0, 'n_estimators': 500, 'learning_rate': 0.3, 'max_depth': 6, 'colsample_bytree': 0.4}

#best_params = {'min_child_weight': 0.5, 'gamma': 0, 'subsample': 0.8, 'objective': 'reg:squarederror', 'max_depth': 3, 'silent': False, 'reg_lambda': 100.0, 'n_estimators': 500, 'colsample_bylevel': 0.8, 'colsample_bytree': 1.0, 'learning_rate': 0.01}

#best_params = {'subsample': 1.0, 'silent': False, 'gamma': 1.0, 'reg_lambda': 100.0, 'min_child_weight': 0.5, 'objective': 'reg:squarederror', 'learning_rate': 0.3, 'max_depth': 2, 'colsample_bytree': 0.8, 'n_estimators': 100, 'colsample_bylevel': 0.5}

#with open('hypsearch_spk.model', 'wb') as f:
#    pickle.dump(ret, f)