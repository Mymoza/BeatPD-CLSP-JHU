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
results = []
#baselines = []

preds = []
results_per_patient = {}
print('ALLLLLLLO 1')
print('len(sys.argv) : ', len(sys.argv))
for i in range(5):
    print(' i value ! : ', i)
    test = pd.read_csv(sys.argv[i]).squeeze()
    w1, w2 = [], []
    tot = 0
    print('spks : ', spks)
    for spk in spks:
        ids = int(spk)
        al_spk = al.loc[al['subject_id'] == spk]
        idx = al_spk['measurement_id'].isin(test)

        #tr_w = al[~idx].groupby('subject_id').count().reset_index()[["subject_id", obj]].rename(columns={obj: 'spcount'})
        #tr = pd.merge(al[~idx], tr_w, on='subject_id')
        tr = al_spk[~idx].drop(['fold_id'], axis=1)
        tr_w = tr['spcount'] ** -0.5
        tr_y = tr[obj].astype(pd.np.float32)
        tr = tr.drop([obj, 'subject_id', 'measurement_id', 'spcount'], axis=1).astype(pd.np.float32)

        #te_w = al[idx].groupby('subject_id').count().reset_index()[["subject_id", obj]].rename(columns={obj: 'spcount'})
        #te = pd.merge(al[idx], te_w, on='subject_id')
        te = al_spk[idx].drop(['fold_id'], axis=1)
        tot += te.shape[0]
        te_w = te['spcount'] ** -0.5
        te_y = te[obj].astype(pd.np.float32)
        sub = te['subject_id']
        sid = te.subject_id
        tid = te.measurement_id
        te = te.drop([obj, 'subject_id', 'measurement_id', 'spcount'], axis=1).astype(pd.np.float32)

        clf = xgb.XGBRegressor(**cfgs[ids])
        #clf = xgb.XGBClassifier(**params)
        clf.fit(
            tr, tr_y,
            sample_weight=tr_w,
            eval_set=[(tr, tr_y)],
            #eval_set=[(tr, tr_y), (te, te_y)],
            #eval_metric=',
            sample_weight_eval_set=[tr_w],
            #sample_weight_eval_set=[tr_w, te_w],
            verbose=0,
            early_stopping_rounds=100
        )
        #best_itr = clf.get_booster().best_iteration
        #clf = xgb.XGBRegressor(**cfgs[ids])
        #clf.fit(
        #    tr, tr_y,
        #    sample_weight=tr_w,
        #    #eval_set=[(tr, tr_y)],
        #    eval_set=[(tr, tr_y), (te, te_y)],
        #    #eval_metric=',
        #    #sample_weight_eval_set=[tr_w],
        #    sample_weight_eval_set=[tr_w, te_w],
        #    verbose=0,
        #    early_stopping_rounds=100
        #)
        #print("Best Iteration: {} {} {}".format(best_itr, clf.get_booster().best_iteration, clf.get_booster().best_score))
        pred = clf.predict(te).clip(0, 4)
        mse = (pred - te_y) ** 2
        w1.append((mse * te_w).sum().squeeze())
        w2.append(te_w.sum().squeeze())
        #print("{0} {1}".format(ids, mse.mean()))
        if ids not in results_per_patient:
            results_per_patient[ids] = []
        results_per_patient[ids].append(mse.mean())
        res = pd.DataFrame(data={'measurement_id': tid, 'subject_id': sid, obj: pred})
        res = pd.merge(res, avg, on='subject_id')
        res[obj] += res['sp_' + obj]
        res = res[["measurement_id", obj]]
        preds.append(res)
    print("ALLLLLLLLLLLOOOOOOOOOOOOO")
    print("{0} {1}".format(tot, al.shape[0]))
    results.append(sum(w1) / sum(w2))
    #mse = te_y.to_numpy() ** 2
    #mse2 = te_y ** 2
    #ret = pd.concat([sub, mse], axis=1)
    #ret.to_csv('tmp.csv')
    #mse = (ret.groupby('subject_id').mean())[obj].to_numpy()
    #cnt = (ret.groupby('subject_id').count())[obj].to_numpy()
    #cnt = cnt ** 0.5
    #results.append(((mse * te_w).sum() / te_w.sum()).squeeze())
    #baselines.append(((mse2 * te_w).sum() / te_w.sum()).squeeze())

# MARIE COMMENT 
### preds = pd.concat(preds)
###preds.to_csv('kfold_prediction_{0}.csv'.format(obj), index=False)

#print(clf.get_booster().get_score(importance_type='gain'))

for i in results_per_patient:
    print("{0}:{1}".format(i, sum(results_per_patient[i])/len(results_per_patient[i])))
print(np.mean(results))
#print(np.mean(baselines))
#plot_importance(clf,max_num_features=10)
#plt.savefig('importance.png')
