import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import sys
import xgboost as xgb
from xgboost import plot_importance
import numpy as np
import pickle

# subchallenge
obj = sys.argv[1]
all_obj = ["on_off", "tremor", "dyskinesia"]

# Training features extracted 
all_fea = pd.read_csv(sys.argv[2])

# Labels from all folds
all_lab = pd.read_csv(sys.argv[3])
all_lab = all_lab.drop(list(set(all_obj) - set([obj])), axis=1)

# Merge features and labels based on their measurement id 
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
subject_id = pd.get_dummies(al.subject_id, columns='subject_id', prefix='spk_')
#al = pd.merge(al, mean_value, on='subject_id')
al = pd.concat([al, subject_id], axis=1)
#al = al.astype(pd.np.float32)

Y = al[obj].to_numpy()
W = al['spcount'].to_numpy() ** -0.5
foldid = al['fold_id'].to_numpy().astype(int)
from sklearn.model_selection import PredefinedSplit
cv = PredefinedSplit(foldid)

X = al.drop([obj, 'subject_id', 'measurement_id', 'spcount', 'fold_id'], axis=1).astype(pd.np.float32).to_numpy()

best_params = {'subsample': 1.0, 'silent': False, 'gamma': 1.0, 'reg_lambda': 100.0, 'min_child_weight': 0.5, 'objective': 'reg:squarederror', 'learning_rate': 0.3, 'max_depth': 2, 'colsample_bytree': 0.8, 'n_estimators': 100, 'colsample_bylevel': 0.5}

# test features
test_fea = pd.read_csv(sys.argv[4])

# Test Data Labels (measurement_id, subject_id)
test_lab = pd.read_csv(sys.argv[5]) #spk label
test_fea = pd.merge(test_fea, test_lab, on=["measurement_id"])
test_sub = pd.read_csv(sys.argv[6]) #submission file
test_fea = pd.merge(test_fea, test_sub, on=['measurement_id'], how='inner')
subject_id = pd.get_dummies(test_fea.subject_id, columns='subject_id', prefix='spk_')
test_fea = pd.concat([test_fea, subject_id], axis=1)


tr_w = al['spcount'] ** -0.5
tr_y = al[obj].astype(pd.np.float32)
tr_id = al.measurement_id
print('tr_id : ', tr_id)
tr_sid = al.subject_id
print('tr_sid : ', tr_sid)
tr = al.drop([obj, 'subject_id', 'measurement_id', 'spcount', 'fold_id'], axis=1).astype(pd.np.float32)

te = pd.merge(test_fea, avg, on='subject_id')
for i in remove:
    if i[3:] != obj: # don't have obj
        te[i[3:]] = te[i[3:]] - te[i]
te = te.drop(remove, axis=1)
te_id = te.measurement_id
te_sid = te.subject_id
te_bak = te.drop(['measurement_id', 'subject_id', 'prediction'], axis=1)#.astype(pd.np.float32)

preds_train_folds = []
preds = []
for i in range(5):
    #test = pd.read_csv(sys.argv[i]).squeeze()
    #idx = al['measurement_id'].isin(test)
    idx = al['fold_id'] == i

    #tr_w = al[~idx].groupby('subject_id').count().reset_index()[["subject_id", obj]].rename(columns={obj: 'spcount'})
    #tr = pd.merge(al[~idx], tr_w, on='subject_id')
    tr = al[~idx].drop(['fold_id'], axis=1)
    tr_w = tr['spcount'] ** -0.5
    tr_y = tr[obj].astype(pd.np.float32)
    tr = tr.drop([obj, 'subject_id', 'measurement_id', 'spcount'], axis=1).astype(pd.np.float32)

    #te_w = al[idx].groupby('subject_id').count().reset_index()[["subject_id", obj]].rename(columns={obj: 'spcount'})
    #te = pd.merge(al[idx], te_w, on='subject_id')
    te = al[idx].drop(['fold_id'], axis=1)
    te_w = te['spcount'] ** -0.5
    te_y = te[obj].astype(pd.np.float32)
    sub = te['subject_id']
    te = te.drop([obj, 'subject_id', 'measurement_id', 'spcount'], axis=1).astype(pd.np.float32)

    clf = xgb.XGBRegressor(**best_params)
    #clf = xgb.XGBClassifier(**params)
    clf.fit(
        tr, tr_y,
        sample_weight=tr_w,
        eval_set=[(tr, tr_y), (te, te_y)],
        #eval_metric=',
        sample_weight_eval_set=[tr_w, te_w],
        verbose=0,
        early_stopping_rounds=100
    )
    #pred = clf.predict(te).clip(0, 4)
    #mse = (pred - te_y) ** 2
    #mse = te_y.to_numpy() ** 2
    #mse2 = te_y ** 2
    #ret = pd.concat([sub, mse], axis=1)
    #ret.to_csv('tmp.csv')
    #mse = (ret.groupby('subject_id').mean())[obj].to_numpy()
    #cnt = (ret.groupby('subject_id').count())[obj].to_numpy()
    #cnt = cnt ** 0.5
    pred = clf.predict(te_bak).clip(0, 4)
    preds.append(pred)
    print('len(tr) : ', len(tr))
    pred_train_fold = clf.predict(tr).clip(0,4)
    preds_train_folds.append(pred_train_fold)
preds = np.array(preds)
preds_train_folds = np.array(preds_train_folds)
print(preds_train_folds[0])

print('-------------')
print(preds_train_folds[1])
ivg = al.groupby('subject_id').mean().reset_index().add_prefix('sp_').rename(columns={'sp_subject_id':'subject_id'})
# Predictions on test set
#print(preds.std(axis=0).max())
print('len(te_id) : ', len(te_id))
print('len(te_sid) : ', len(te_sid))
print('len(preds) : ', len(preds)) 
print('len(preds[0] : ', len(preds[0])) 
print('len(preds[1] : ', len(preds[1])) 
res = pd.DataFrame(data={'measurement_id': te_id, 'subject_id': te_sid, obj: preds.mean(axis=0)})
res = pd.merge(res, avg, on='subject_id')
res[obj] += res['sp_' + obj]
res = res[["measurement_id", obj]]
#print(res)
res.to_csv('submission/cis-pd.{0}_new.csv'.format(obj), index=False)

# FIXME: Predictions on training folds
print('len(tr_id) : ', len(tr_id))
print('len(tr_sid) : ', len(tr_sid))
print('len(preds_train_folds) : ', len(preds_train_folds))
results_folds = pd.DataFrame(data={'measurement_id': tr_id, 'subject_id': tr_sid, obj: preds_train_folds.mean(axis=0)})
print(results_folds)
print('results_folds len : ', len(results_folds))
#results_folds = pd.merge(results_folds, avg, on='subject_id')
preds.to_csv('submission/kfold_prediction_cis-pd_{0}.csv'.format(obj), index=False)
print('preds.to_csv kfold_prediction_cis-pd was done')
#print(clf.get_booster().get_score(importance_type='gain'))
