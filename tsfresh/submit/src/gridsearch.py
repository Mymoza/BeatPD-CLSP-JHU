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
import argparse
from sklearn.model_selection import KFold
from lofo import LOFOImportance, Dataset, plot_importance, FLOFOImportance
from sklearn.model_selection import PredefinedSplit

# on_off features/cis-pd.training.csv ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv 1.0
parser = argparse.ArgumentParser(description='Perform gridsearch or predicts on test folds.')
parser.add_argument('symptom', metavar='obj', type=str, help='Should be either on_off, tremor, or dyskinesia')
parser.add_argument("--features", action="append", type=str, help='Path to the features, like features/cis-pd.training.csv')
parser.add_argument("--labels", type=str, help='Path to the labels, for example \{\path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv')
parser.add_argument("--linear_combination", type=float, help='Path to the labels, for example')
parser.add_argument("--filename", type=str, help='filename')
parser.add_argument("--load_model", type=str, help='filename') #FIXME: It's a flag but as well as its not none will work
parser.add_argument("--random_forest", type=str, help='filename') #FIXME: Flag too. If it's there, then we want to use RFR instead of xgboost
parser.add_argument("--find_best_params", type=str, help='filename') #FIXME: flag
parser.add_argument("--pred_path", type=str, help='path to pred files')
parser.add_argument("--msek_path", type=str, help='path to save msek file (mse per subject per fold)')
parser.add_argument("--lofo_importance", type=str, help='flag if you want lofo feature importance') #FIXME flag

args = parser.parse_args()

# contains the subchallenge we are working on 
obj = args.symptom

# All the subchallenges
all_obj = ["on_off", "tremor", "dyskinesia"]

# Read the training features, we can provide more than one file (for data augmentation purposes)
# all_features = pd.read_csv(args.features[0])
# all_features = pd.concat([all_features, pd.read_csv(args.features[1], header=0)])
all_features = pd.concat((pd.read_csv(f) for f in args.features))
print('all_features.shape : ', all_features.shape)
# Read the training labels
all_labels = pd.read_csv(args.labels)
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

# squareroot(1/number of recordings) 
W = np.array(all_features_labels['spcount']) ** -0.5
foldid = np.array(all_features_labels['fold_id']).astype(int)
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
    'n_estimators': [50, 65, 95, 100, 500, 1000],
    #'n_estimators': [50, 55, 63, 64, 65, 67, 71, 74, 75, 78, 83, 86, 93, 95, 96, 97, 100],# 500, 1000],
    'max_depth': [2, 3, 4, 5, 6, 8, 15, 25, 30],
    'min_samples_split': [2, 5, 10, 15, 100],
    'min_samples_leaf': [1, 2, 5, 10],
    #'max_features': ['auto','0.25']
}

# FIXME: Uncomment this section to perform the gridsearch 
from sklearn.model_selection import RandomizedSearchCV

if args.random_forest:
    # Random Forest Regressor 
    clf = RandomForestRegressor()
    
else:
    # xgboost
    clf = xgb.XGBRegressor()

if args.random_forest and args.find_best_params:
    rs_clf = RandomizedSearchCV(clf, param_grid, n_iter=100,
                                n_jobs=8, verbose=2, cv=cv,
                                refit=False, random_state=42, scoring='neg_mean_squared_error')
    rs_clf.fit(X, Y, sample_weight=W) 
    best_params = rs_clf.best_params_
    print('Best Params : ')
    print(best_params)
elif args.random_forest:
    # For the random Forest regressor
    best_params = {'max_depth': 2, 'n_estimators': 100}
else:   
    # Best parameters used for submission 3 with the xgboost
    best_params = {'subsample': 1.0, 'silent': False, 'gamma': 1.0, 'reg_lambda': 100.0, 'min_child_weight': 0.5, 'objective': 'reg:squarederror', 'learning_rate': 0.3, 'max_depth': 2, 'colsample_bytree': 0.8, 'n_estimators': 100, 'colsample_bylevel': 0.5}


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

# Will be a float value of the lambda if provided, otherwise None 
lambda_value = args.linear_combination

all_spks = all_features_labels['subject_id'].unique()

columns_to_exclude = (all_features_labels.columns[(all_features_labels.dtypes.values != np.dtype('float64'))]).to_list()
columns_to_exclude.remove('fold_id')

for i in range(5):
    ##test = pd.read_csv(sys.argv[i]).squeeze()
    ##idx = all_features_labels['measurement_id'].isin(test)

    # Filter all_features_labels to only the measurements of the current fold
    idx = all_features_labels['fold_id'] == i

    ##tr_w = all_features_labels[~idx].groupby('subject_id').count().reset_index()[["subject_id", obj]].rename(columns={obj: 'spcount'})
    ##tr = pd.merge(all_features_labels[~idx], tr_w, on='subject_id')
    
    # Drop the measurements that are not from the current fold 
    tr = all_features_labels[~idx].drop(['fold_id'], axis=1)

    # Data augmentation with a lambda
    if lambda_value is not None:
        for spk in all_spks: 
            # Filter training data for that speaker 
            tr_subject = tr.loc[tr['subject_id'] == spk]
            
            # FIXME: The mean number of recordings changes per fold so to make it easier 
            # I'm just gonna use the mean for the data augmented values
            mean_spcount = int(tr_subject['spcount'].mean())

            # Apply the lambda on the training features except the columns that are not float
            modDfObj1 = tr_subject[tr_subject.columns.difference(columns_to_exclude)].reset_index(drop=True).apply(lambda x: x * lambda_value, axis=1, result_type='broadcast')
            modDfObj2 = tr_subject[tr_subject.columns.difference(columns_to_exclude)].sample(frac=1).reset_index(drop=True).apply(lambda x: x * (1-lambda_value), axis=1, result_type='broadcast')
            
            df_data_aug = []

            for index, measurement1 in modDfObj1.iterrows():
                measurement2 = modDfObj2.loc[index]
                index2 = 1
                while measurement1[obj] * lambda_value == measurement2[obj] * (1-lambda_value):
                    measurement2 = modDfObj2.loc[(index+index2 if (index+index2<len(modDfObj2)) else 0+index2)]
                    index2 = index2 + 1 
                    if index2 == len(modDfObj2): #some folds only contain the same label for a speaker
                        break
                df_data_aug.append(measurement1.add(measurement2).to_list())
            # First loop to go over the rows
            # for index, measurement1 in modDfObj1.iterrows():
            #     # Second rows to go over the loop except the same two rows and do not repeat same substraction
            #     for index2, measurement2 in modDfObj2.iterrows():
            #         if measurement1[obj] * lambda_value == measurement2[obj] * (1-lambda_value):
            #             print('Labels were : ', str((measurement1[obj] * lambda_value)), ' and ', str((measurement2[obj] * (1-lambda_value))))
            #             continue
            #         df_data_aug.append(measurement1.add(measurement2).to_list())
            df_data_aug = pd.DataFrame(df_data_aug, columns=modDfObj1.columns)

            # Add again the columns we just removed 
            #modDfObj1 = pd.concat([modDfObj1, tr_subject[columns_to_exclude]], axis=1)
            subjects_columns_title = [x for x in columns_to_exclude if x.startswith("spk_")]
            # Select the first row of spk_ elements 
            subject_row = tr_subject.iloc[0,tr_subject.columns.str.startswith("spk_")]
            subject_row = subject_row.append(pd.Series(mean_spcount, index=["spcount"]))
            # Duplicate that first row for the number of elements we need
            # full_subjects = pd.concat([subject_row]*len(df_data_aug), ignore_index=True)
            #FIXME NEEDED?
            # full_subjects = pd.DataFrame(full_subjects, columns=subjects_columns_title)

            full_subjects = pd.concat([pd.DataFrame(subject_row).T]*len(df_data_aug), ignore_index=True)
            df_data_aug = pd.concat([df_data_aug, full_subjects], axis=1)
            #df_data_aug = pd.concat([df_data_aug, full_subjects], axis=1)

            # Append the augmented dataframe to the original dataframe
            print('Before spk ', str(spk), ' tr shape is : ', tr.shape)
            tr = pd.concat([tr, df_data_aug], ignore_index=True)
            print('After spk ', str(spk), ' tr shape is : ', tr.shape)
            
    # print('obj is : ', obj)
    train_y = tr[obj].astype(pd.np.float32) # training labels 
    # print('tr spcount : ', tr['spcount'])
    train_weight = tr['spcount'] ** -0.5 # training weight
    tr = tr.drop([obj, 'subject_id', 'measurement_id', 'spcount'], axis=1).astype(pd.np.float32)

    ##te_w = all_features_labels[idx].groupby('subject_id').count().reset_index()[["subject_id", obj]].rename(columns={obj: 'spcount'})
    ##te = pd.merge(all_features_labels[idx], te_w, on='subject_id')
    
    # Drop the measurements that are used in the training of this fold, so we keep [idx] instead of [~idx]
    te = all_features_labels[idx].drop(['fold_id'], axis=1)
    test_weight = te['spcount'] ** -0.5 # test weight 
    test_y = te[obj].astype(pd.np.float32) # testing labels
    sid = te.subject_id
    test_measurement_id = te.measurement_id
    te = te.drop([obj, 'subject_id', 'measurement_id', 'spcount'], axis=1).astype(pd.np.float32)

    if args.random_forest:
        # Random Forest Regressor 
        print('Using Random Forest Regressor')
        clf = RandomForestRegressor(**best_params)
        clf.fit(tr, train_y, sample_weight=train_weight)
    else:
        # XGBoost 
        print('Using XGBOOST')
        clf = xgb.XGBRegressor(**best_params)
        #clf = xgb.XGBClassifier(**params)

        # Fit for the xgboost 
        clf.fit(
            tr, train_y,
            sample_weight=train_weight,
            eval_set=[(tr, train_y), (te, test_y)],
            #eval_metric=',
            sample_weight_eval_set=[train_weight, test_weight],
            verbose=0,
            early_stopping_rounds=100
        )
    pred = clf.predict(te).clip(0, 4)
    # Computes the Msek 
    mse = (pred - test_y) ** 2
    # Computes the Msek for the null model 
    mse2 = test_y ** 2
    # mean_y = labels of that speaker.sum() / spcount of that subject 
    # mse_null_model = (mean_y - test_y) ** 2

    # Code to get the MSEK as per the challenge requirement 
    # ret = pd.concat([sub, mse], axis=1)
    # mse_marie = (ret.groupby('subject_id').sum())[obj].to_numpy()

    #ret.to_csv('tmp.csv')
    
    #mse = (ret.groupby('subject_id').mean())[obj].to_numpy()
    #cnt = (ret.groupby('subject_id').count())[obj].to_numpy()
    
    #cnt = cnt ** 0.5
    res = pd.DataFrame(data={'measurement_id': test_measurement_id, 'subject_id': sid, obj: pred})
    res = pd.merge(res, avg, on='subject_id')
    res[obj] += res['sp_' + obj]
    res = res[["measurement_id", obj]]
    preds.append(res)

    # Computes the Final Score (formula from the challenge)
    results.append(((mse * test_weight).sum() / test_weight.sum()).squeeze())
    baselines.append(((mse2 * test_weight).sum() / test_weight.sum()).squeeze())
preds = pd.concat(preds)
pred_path = args.pred_path if args.pred_path is not None else ''
print('Saving preds to path : ', pred_path)

# Save the MSE per subject per fold to a file so we can do statistical analysis 
msek_path = args.msek_path if args.msek_path is not None else None
print('msek_path is : ', msek_path)
results = pd.DataFrame(results)

if args.filename:
     preds.to_csv(pred_path+'kfold_prediction_cis-pd_{0}_{1}.csv'.format(obj, args.filename), index=False)
else:
    preds.to_csv(pred_path+'kfold_prediction_cis-pd_{0}.csv'.format(obj), index=False)

#print(clf.get_booster().get_score(importance_type='gain'))
if msek_path is not None:
    # df_preds = pd.concat(preds)
    true_labels = all_labels[["measurement_id", obj, 'subject_id']]
    true_labels = true_labels.dropna(subset=[obj])
    true_labels = true_labels.rename(columns={obj: 'true_'+obj})
    # DataFrame with measurement_id, obj prediction, true label, subject_id 
    preds_true_labels = pd.merge(preds, true_labels, on='measurement_id')

    msek_subject = []
    for spk in all_spks: 
        spk_preds_true_label = preds_true_labels[preds_true_labels['subject_id'] == spk]
        msek = ((spk_preds_true_label['true_'+obj] - spk_preds_true_label[obj]) ** 2).sum() * (1/len(spk_preds_true_label))
        print(spk , ' : ', msek)
        msek_subject.append(msek)

    df_msek_subject = pd.DataFrame(msek_subject)
    df_msek_subject.to_csv(msek_path+'msek_cis-pd_{0}_{1}.csv'.format(obj, args.filename), index=False, header=False)

print("baseline {0} result {1}".format(np.mean(baselines),np.mean(results)))
if not args.random_forest and msek_path is not None:
    fig, ax = plt.subplots(figsize=(19, 10))
    xgb.plot_importance(clf, ax=ax, importance_type='gain', max_num_features=20, xlabel='Feature importance score')
    plt.savefig(msek_path+'importance_{0}_{1}_gain.png'.format(obj, args.filename), format='png')
    plt.savefig(msek_path+'importance_{0}_{1}_gain.pdf'.format(obj, args.filename), format='pdf')

    # Flag to know if we want to compute lofo importance 
    lofo_importance = args.lofo_importance if args.lofo_importance is not None else None 
    print('lofo importance : ', lofo_importance)
    if lofo_importance is not None:
        target = obj #target will be either tremor, dyskinesia, or on_off 
        dataset = Dataset(df=all_features_labels, target=target, features=[col for col in all_features_labels.columns if col != target])

        # define the validation scheme and scorer. The default model is LightGBM
        print('te val : ', te)
        print('te.to_list : ', te.columns)
        
        # Tried Fast LOFO Importance, but the validation set is not big enough
        #lofo_imp = FLOFOImportance(trained_model=clf, validation_df=te, features=list(te.columns), target=target, scoring="neg_mean_squared_error") 
        lofo_imp = LOFOImportance(dataset, cv=cv, scoring="neg_mean_squared_error")
        # get the mean and standard deviation of the importances in pandas format
        importance_df = lofo_imp.get_importance()
        # plot the means and standard deviations of the importances

        figsize=(12, 25)
        # figsize=(20,10)
        kind="default"

        # I put the plot_importance function here just to save the actual figure 
        importance_df = importance_df.copy()
        importance_df["color"] = (importance_df["importance_mean"] > 0).map({True: 'g', False: 'r'})
        importance_df.sort_values("importance_mean", inplace=True)

        available_kinds = {"default", "box"}
        if kind not in available_kinds:
            warnings.warn("{kind} not in {ak}. Setting to default".format(kind=kind, ak=available_kinds))

        if kind == "default":
            importance_df.plot(x="feature", y="importance_mean", xerr="importance_std",
                            kind='barh', color=importance_df["color"], figsize=figsize)
            plt.savefig(msek_path+'lofo_importance_{0}_{1}.png'.format(obj, args.filename))
            
        # elif kind == "box":
        #     lofo_score_cols = [col for col in importance_df.columns if col.startswith("val_imp")]
        #     features = importance_df["feature"].values.tolist()
        #     importance_df.set_index("feature")[lofo_score_cols].T.boxplot(column=features, vert=False, figsize=figsize)


