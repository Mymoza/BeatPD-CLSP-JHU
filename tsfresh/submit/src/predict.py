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
subject_id = pd.get_dummies(al.subject_id, columns='subject_id', prefix='spk_')
al = pd.concat([al, subject_id], axis=1)

test_fea = pd.read_csv(sys.argv[4])
test_lab = pd.read_csv(sys.argv[5]) #spk label
test_fea = pd.merge(test_fea, test_lab, on=["measurement_id"])
test_sub = pd.read_csv(sys.argv[6]) #submission file
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
te = te.drop(['measurement_id', 'subject_id', 'prediction'], axis=1).astype(pd.np.float32)

import pickle
with open('mdl/cis-pd.conf','rb') as f:
    params=pickle.load(f)
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
results = pd.DataFrame(data={obj: clf.predict(te).clip(0,4)})
joined = pd.DataFrame(te_id).join(results).join(te_sid)
joined = pd.merge(joined, avg, on='subject_id')
joined[obj] += joined['sp_' + obj]
joined = joined[['measurement_id', obj]]
joined.to_csv('submission/cis-pd.{0}.csv'.format(obj), index=False)
#ret = pd.concat([sub, mse], axis=1)
#ret.to_csv('tmp.csv')
#mse = (ret.groupby('subject_id').mean())[obj].to_numpy()
#cnt = (ret.groupby('subject_id').count())[obj].to_numpy()
#cnt = cnt ** 0.5
