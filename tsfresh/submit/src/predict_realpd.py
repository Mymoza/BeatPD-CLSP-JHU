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
subject_id = pd.get_dummies(al.subject_id, columns='subject_id', prefix='spk_')
al = pd.concat([al, subject_id], axis=1)

test_fea = pd.read_csv(sys.argv[4])
test_lab = pd.read_csv(sys.argv[5]) #spk label
test_fea = pd.merge(test_fea, test_lab, on=["measurement_id"])
test_sub = pd.read_csv(sys.argv[6]) #submission file
test_fea = pd.merge(test_fea, test_sub, on=['measurement_id'], how='inner')
subject_id = pd.get_dummies(test_fea.subject_id, columns='subject_id', prefix='spk_')
test_fea = pd.concat([test_fea, subject_id], axis=1)

#params = {
#    "learning_rate": 0.05,
#    "max_depth": 3,
#    "n_estimators": 1000,
#    "min_child_weight": 1,
#    "colsample_bytree": 0.9,
#    "subsample": 0.8,
#    "nthread": 12,
#    "random_state": 42,
#    #"objective": "multi:softprob",
#    #"num_class": 5
#    "objective": "reg:squarederror",
#}

import pickle
with open('mdl/real-pd.conf', 'wb') as f:
    params=pickle.load(f)

avg = al.groupby('subject_id').mean().reset_index().add_prefix('sp_').rename(columns={'sp_subject_id':'subject_id'})
tr = pd.merge(al, avg, on='subject_id')
remove = []
for i in tr.columns:
    if not i.startswith('sp_') and 'sp_' + i in tr.columns:
        remove.append('sp_' + i)
        tr[i] = tr[i] - tr['sp_' + i]
tr = tr.drop(remove, axis=1)

tr_w = al.groupby('subject_id').count().reset_index()[["subject_id", obj]].rename(columns={obj: 'spcount'})
tr = pd.merge(tr, tr_w, on='subject_id')
tr_w = tr['spcount'] ** -0.5
tr_y = tr[obj].astype(pd.np.float32)
tr = tr.drop([obj, 'subject_id', 'measurement_id', 'spcount'], axis=1).astype(pd.np.float32)

te = pd.merge(test_fea, avg, on='subject_id')
for i in remove:
    if i[3:] != obj: # don't have obj
        te[i[3:]] = te[i[3:]] - te[i]
te = te.drop(remove, axis=1)
te_id = te.measurement_id
te_sid = te.subject_id
te = te.drop(['measurement_id', 'subject_id', 'prediction'], axis=1).astype(pd.np.float32)

clf = xgb.XGBRegressor(**params)
#clf = xgb.XGBClassifier(**params)
clf.fit(
    tr, tr_y,
    sample_weight=tr_w,
    eval_set=[(tr, tr_y)],
    #eval_metric=',
    sample_weight_eval_set=[tr_w],
    verbose=0,
    early_stopping_rounds=100
)
results = pd.DataFrame(data={obj: clf.predict(te).clip(0,4 if obj != 'on_off' else 1)})
joined = pd.DataFrame(te_id).join(results).join(te_sid)
joined = pd.merge(joined, avg, on='subject_id')
joined[obj] += joined['sp_' + obj]
joined = joined[['measurement_id', obj]]
joined.to_csv('submission/{0}.{1}.csv'.format(sys.argv[7], obj), index=False)
#ret = pd.concat([sub, mse], axis=1)
#ret.to_csv('tmp.csv')
#mse = (ret.groupby('subject_id').mean())[obj].to_numpy()
#cnt = (ret.groupby('subject_id').count())[obj].to_numpy()
#cnt = cnt ** 0.5
