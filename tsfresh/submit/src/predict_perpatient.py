import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import sys
import xgboost as xgb
from xgboost import plot_importance
import numpy as np

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
al = pd.merge(al, pd.read_csv('data/order.csv'), how='inner', on=["measurement_id"])
weight = al.groupby(['subject_id', 'fold_id']).count().reset_index()[["subject_id", "fold_id", obj]].rename(columns={obj: 'spcount'})
al = pd.merge(al, weight, on=['subject_id', 'fold_id'])

test_fea = pd.read_csv(sys.argv[4])
test_lab = pd.read_csv(sys.argv[5]) #spk label
test_fea = pd.merge(test_fea, test_lab, on=["measurement_id"])
test_sub = pd.read_csv(sys.argv[6]) #submission file
test_fea = pd.merge(test_fea, test_sub, on=['measurement_id'], how='inner')


#tr = al.drop([obj, 'subject_id', 'measurement_id', 'spcount', 'fold_id'], axis=1).astype(pd.np.float32)

te = pd.merge(test_fea, avg, on='subject_id')
for i in remove:
    if i[3:] != obj: # don't have obj
        te[i[3:]] = te[i[3:]] - te[i]
te = te.drop(remove, axis=1)
#te = te.drop(['measurement_id', 'subject_id', 'prediction'], axis=1).astype(pd.np.float32)
all_spks = al['subject_id'].unique()

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

A = []
for spk in all_spks:
    spk_id = int(spk)
    tr_spk = al.loc[al['subject_id'] == spk]
    tr_y = tr_spk[obj].astype(pd.np.float32)
    tr_w = tr_spk['spcount'] ** -0.5
    tr_spk = tr_spk.drop([obj, 'subject_id', 'measurement_id', 'spcount', 'fold_id'], axis=1).astype(pd.np.float32)

    te_spk = te.loc[te['subject_id'] == spk]
    te_id = te_spk.measurement_id.reset_index(drop=True)
    te_sid = te_spk.subject_id.reset_index(drop=True)
    te_spk = te_spk.drop(['measurement_id', 'subject_id', 'prediction'], axis=1).astype(pd.np.float32)

    #import pickle
    #with open('mdl/cis-pd.'+obj+'.'+str(spk)+'.conf','rb') as f:
    #    params=pickle.load(f)
    #print('cfgs['+str(spk)+'] = ', params)    
    
    # Using the params saved to file in the gridsearch_perpatient
    #clf = xgb.XGBRegressor(**params) 

    # Using the saved configuration provided at the top of this document
    clf = xgb.XGBRegressor(**cfgs[spk_id])
    
    # Currently Using Stop Criteria on Training Data for the speaker
    clf.fit(
        tr_spk, tr_y,
        sample_weight=tr_w,
        eval_set=[(tr_spk, tr_y)],
        #eval_metric=',
        sample_weight_eval_set=[tr_w],
        verbose=0,
        early_stopping_rounds=100
    )
    results = pd.DataFrame(data={obj: clf.predict(te_spk).clip(0,4)})
    joined = pd.DataFrame(te_id).join(results).join(te_sid)
    joined = pd.merge(joined, avg, on='subject_id')
    joined[obj] += joined['sp_' + obj]
    joined = joined[['measurement_id', obj]]
    A.append(joined)
joined = pd.concat(A)
joined.to_csv('submission/cis-pd.{0}.perpatient.csv'.format(obj), index=False)
#ret = pd.concat([sub, mse], axis=1)
#ret.to_csv('tmp.csv')
#mse = (ret.groupby('subject_id').mean())[obj].to_numpy()
#cnt = (ret.groupby('subject_id').count())[obj].to_numpy()
#cnt = cnt ** 0.5
