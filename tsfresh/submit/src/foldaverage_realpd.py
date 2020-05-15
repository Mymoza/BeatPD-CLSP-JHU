import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import sys
import xgboost as xgb
from xgboost import plot_importance
import numpy as np
import pickle

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
al = pd.merge(al, pd.read_csv(sys.argv[4]), how='inner', on=["measurement_id"])
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

#best_params = {'max_depth': 3, 'colsample_bylevel': 0.8, 'colsample_bytree': 0.4, 'silent': False, 'n_estimators': 100, 'learning_rate': 0.3, 'objective': 'reg:squarederror', 'reg_lambda': 50.0, 'min_child_weight': 5.0, 'subsample': 0.7, 'gamma': 1.0}

best_params = {
    "learning_rate": 0.05,
    "max_depth": 3,
    "n_estimators": 1000,
    "min_child_weight": 1,
    "colsample_bytree": 0.9,
    "subsample": 0.8,
    "nthread": 12,
    "random_state": 42,
    #"objective": "multi:softprob",
    #"num_class": 5
    "objective": "reg:squarederror",
}

test_fea = pd.read_csv(sys.argv[5])
test_lab = pd.read_csv(sys.argv[6]) #spk label
test_fea = pd.merge(test_fea, test_lab, on=["measurement_id"])
test_sub = pd.read_csv(sys.argv[7]) #submission file
test_fea = pd.merge(test_fea, test_sub, on=['measurement_id'], how='inner')
subject_id = pd.get_dummies(test_fea.subject_id, columns='subject_id', prefix='spk_')
test_fea = pd.concat([test_fea, subject_id], axis=1)


tr_w = al['spcount'] ** -0.5
tr_y = al[obj].astype(pd.np.float32)
tr = al.drop([obj, 'subject_id', 'measurement_id', 'spcount', 'fold_id'], axis=1).astype(pd.np.float32)

te = pd.merge(test_fea, avg, on='subject_id')
for i in remove:
    if i[3:] != obj: # don't have obj
        te[i[3:]] = te[i[3:]] - te[i]
te = te.drop(remove, axis=1)
te_id = te.measurement_id
te_sid = te.subject_id
te_bak = te.drop(['measurement_id', 'subject_id', 'prediction'], axis=1)#.astype(pd.np.float32)

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
    pred = clf.predict(te_bak).clip(0, 4 if obj!='on_off' else 1)
    preds.append(pred)
preds = np.array(preds)
#print(preds.std(axis=0).max())
res = pd.DataFrame(data={'measurement_id': te_id, 'subject_id': te_sid, obj: preds.mean(axis=0)})
res = pd.merge(res, avg, on='subject_id')
res[obj] += res['sp_' + obj]
res = res[["measurement_id", obj]]
#print(res)
res.to_csv('submission/{0}_{1}_new.csv'.format(sys.argv[-1], obj), index=False)
# FIXME: Bug to create kfold_prediction files
#preds.to_csv('kfold_prediction_real-pd_{0}_new.csv'.format(obj), index=False)
#print(clf.get_booster().get_score(importance_type='gain'))
