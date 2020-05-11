# BEATPD 

This GitHub repository contains the code to reproduce the results obtained by the team JHU-CLSP during the BeatPD challenge. 


# Step-By-Step guide 

This step-by-step guide will cover the following steps: 

1. [Prepare the data](#1-prepare-data)
2. [Embeddings](#2-embeddings)
    1. [MFCC](#2.1-mfcc) 
    2. [AutoEncoder (AE)](#2.2-autoencoder)
        1. [Train the AutoEncoder](#2.2.1-train-ae)
        2. [Save AE Features](#2.2.2-get-ae-features)
    3. [Create i-vectors](#2.3-create-ivectors)
    4. [Get results for SVR/SVR Per Patient/SVR Everyone](#2.4-get-results)
    5. [Get predictions CSV](#2.5-get-predictions)
3. [TSFRESH + XGBOOST](#3-tsfresh)
4. [Fusion](#4-fusion)

## Set up the environment : 
```
$ conda create -n BeatPD python=3.5
$ source activate BeatPD 
$ conda install --file requirements.txt
```

Make sure that the Jupyter notebook is running on `BeatPD` kernel. 

If the conda environment isn't showing in Jupyter kernels (Kernel > Change Kernel > BeatPD), run: 
```
$ ipython kernel install --user --name=BeatPD
```
You will then be able to select `BeatPD` as your kernel. 

<a name="prepare-data"></a>
## 1. Prepare the data 

All the steps to prepare the data is done in the Jupyter Notebook `prepare_data.ipynb`. 

1. Open the notebook
2. Change the `data_dir` variable for the absolute path to the folder that contains the data given by the challenge. In this folder, you should already have the following directories downloaded from the [challenge website](https://www.synapse.org/#!Synapse:syn20825169/wiki/596118): 
```
/export/b19/mpgill/BeatPD_data  $ ls
cis-pd.ancillary_data  cis-pd.testing_data   real-pd.ancillary_data  real-pd.testing_data
cis-pd.data_labels     cis-pd.training_data  real-pd.data_labels     real-pd.training_data
```
3. Execute the cells in the Notebook. It will create several folders needed to reproduce the experiments. The [data directory structure is documented in the wiki](https://github.com/Mymoza/BeatPD-CLSP-JHU/wiki/1-Data-Directory-Structure).

<a name="embeddings"></a>
## 2. Embeddings 

<a name="2.1-mfcc"></a>
### 2.1 MFCC 

🙅‍♀️: This section hasn't been written yet. It is not a priority as MFCCs did not provide best results and they were not used for submission. 

<a name="2.2-autoencoder"></a>
### 2.2 AutoEncoder (AE) features 

<a name="2.2.1-train-ae"></a>
#### 2.2.1 Train the AutoEncoder 

1. At the moment, all the code needed for the AE lives [on a branch](https://github.com/Mymoza/BeatPD-CLSP-JHU/pull/14). So the first step is to checkout that branch with `git checkout marie_ml_dl_real`.
2. `conda env create --file environment_ae.yml` : This will create the `keras_tf2` environment you need to run AE experiments.
3. Train an AE model & save their features:
    - For CIS-PD: At line 51 of the `train_AE.py` file, change the `save_dir` path to the directory where you want to store the AE models. 
    - For REAL-PD: At line 53 of the `train_AE_real.py` file, change the `save_dir` path to the directory where you want to store the AE models.
4. Launch the training for the configurations you want. Some examples are available in this wiki page about [Creating AutoEncoder Features](https://github.com/Mymoza/BeatPD-CLSP-JHU/wiki/3--Creating-AutoEncoder-Features). To reproduce the results of submission 4, you will need the following command which uses features of length 60 and 400 as frame length: 

`python train_AE.py --latent_dim 60 -dlP '{"remove_inactivity":"False"}' --saveAEFeats --saveFeatDir "/export/b19/mpgill/BeatPD/AE_60ft_400fl_orig/"`

5. This should create the following file `mlp_encoder_uad_False_ld_60.h5` and the features will be saved in the directory provided with the `--saveFeatDir` flag. 

<a name="2.2.2-get-ae-features"></a>
#### 2.2.2 Get AutoEncoder (AE) Features 

1. `git checkout marie_ml_dl_real`: The code to get features from the AutoEncoder is in another branch. 
2. `cd ml_dl`
3. `source activate keras_tf2`
4. Go to this [wiki page](https://github.com/Mymoza/BeatPD-CLSP-JHU/wiki/3--Creating-AutoEncoder-Features#create-autoencoder-features) that lists many examples of commands you can use the create the required AE features. If you only want to get features without creating models, you need to comment a section of the `train_AE.py` and `train_AE_real.py` files. The section needed to be commented is identified directly in the file.
5. Run the command you are interested in getting!

**Submission 4**
To reproduce it, you will need to create the following:
- `AE_30ft_orig_inactivity_removed`
- `cis_testing_AE_30ft_orig_inactivity_removed`

We made a mistake and although we meant to be using `AE_60ft_400fl_orig` (and `cis_testing_AE_60ft_orig`) as that provided us with better crossvalidation results, we ended up using the wrong features! 

### 2.2.3 Create i-vectors 
 
After creating Autoencoder features or the MFCC, we can create i-vectors. 

You need to have [Kaldi](https://kaldi-asr.org/doc/install.html) installed first. Follow Kaldi's instructions to install. 

The following steps will vary a lot depending on what ivector you want to create. One way to decide which ivector to create is to view the [Google spreadsheet results](https://docs.google.com/spreadsheets/d/11l7S49szMllpebGg2gji2aBea35iqLqO5qrlOBSJnIc/edit?usp=sharing) and find out which result you are interested in replicating. The column "C" has notes for each appropriate cell with the name of the ivector folder we use. You can use the same nomenclature to replicate our experiments. 

🛑TODO: Good vocabulary? 

1. `cd kaldi/egs/` : Change your directory to where you installed Kaldi. 
2. `mkdir beatPDivec; cd beatPDivec` : Create a directory to hold the ivectors. 
3.  `cp path-github-repo/sid_novad/* ../sre08/v1/sid/.` : Copy the `novad.sh` files from the repository to your Kaldi's directory 
4. `mkdir *****` : Create a folder with a meaningful name about the ivectors we want to create. The nomenclature we used to name the ivectors we created was also [documented in the wiki](https://github.com/Mymoza/BeatPD-CLSP-JHU/wiki/4-ivectors-nomenclature). To reproduce the final submission, create `dysk_orig_auto60_400fl`.
5. `cd ****` : Change your directory to the ivector folder you just created 
6. `mkdir data`
7. `cp -rf path-github-repo/beatPDivec/default_data/v2_auto/. .`
8. `cp -rf path-github-repo/beatPDivec/default_data/autoencData/data/dyskinesia/. data/.` : Copy the data for the task. In this case, we used dyskinesia. 
9. `ln -s sid ../../sre08/v1/sid; ln -s steps ../../sre08/v1/steps; ln -s utils ../../sre08/v1/utils` : Create symbolic links
10. `vim runFor.sh`: Edit the following variables:
    - `subChallenge`: use either `onoff`, `tremor`, or `dysk`. 
    - `sDirFeats`: use the absolute path to the AE features you want to use, for example `sDirFeats=/export/b19/mpgill/BeatPD/AE_60ft_400fl_orig` 
11. `qsub -l mem_free=30G,ram_free=30G -pe smp 6 -cwd -e /export/b19/mpgill/errors/errors_trem_auto30_noinact_laureano -o /export/b19/mpgill/outputs/outputs_trem_auto30_noinact_laureano runFor.sh`


### 2.2.4 Get results on test folds

The file `runFor.sh` will create the log files with the results of the experiments you ran. The following section explains how to retrieve those results. If you are looking for more manual way of getting results without running `runFor.sh`, there is some documentation in [this wiki page](https://github.com/Mymoza/BeatPD-CLSP-JHU/wiki/4--Manual-Evaluation-Alternatives).

#### 2.2.4.1 Manually - for one size of ivector 
The following example will retrieve results for the following ivector: `trem_noinact_auto30`.

1. `cd /export/c08/lmorove1/kaldi/egs/beatPDivec/trem_noinact_auto30/exp/`
2. `cd ivec_350` : Then, choose an ivector size 
3. `ls` : 
```
globalAccuPLDA.log : Result for PLDA 
globalAccuKNN.log : Result for KNN
globalAccuSVR_Test.log : Result for SVR 
globalAccuPerPatientSVR_Test.log : Result for Per Patient SVR 
globalAccuEveryoneSVR_Test.log : Result for Everyone SVR
```

#### 2.2.4.2 Extract results for different ivector sizes 

As of now, the automation is present in the `get_excel_results.ipynb`, and just creates a table in Jupyter from which we can copy and paste to Excel or Google spreadsheet:

So far, it was only developed for Per Patient SVR and Everyone SVR results.
For the other back-ends, you still need to get the results by hand like it was explained in the previous section. 

<a name="2.5-get-predictions"></a>
### Get Predictions CSV 

### Per Patient SVR 

#### Option 1 - For the test subset of the challenge 

🛑TODO: Make sure these are the right steps with Laureano 

1. `cd` to the ivector location, for example `cd kaldi/egs/beatPDivec/dysk_orig_auto60_400fl/` was the ivector used for the [4th submission](https://github.com/Mymoza/BeatPD-CLSP-JHU/wiki/0-Write-Up).
2. In the file `local/local/pca_svr_bpd2.sh`, make sure that the flag `--bPatientPredictionsPkl` is added to create pkl files for each subject_id, like this:

```
$cmd $sOut/pca_${iComponents}_svr_${sKernel}_${fCValueStr}_${fEpsilon}Testx.log \
     pca_knn_bpd2.py --input-trai $sFileTrai \
     --input-test $sFileTest \
     --output-file $sOut \
     --iComponents $iComponents \
     --sKernel $sKernel \
     --fCValue $fCValue \
     --fEpsilon $fEpsilon \
     --bPatientPredictionsPkl
conda deactivate
```

3. Run `runFinalsubm3_2.sh`. This will call `run_Final_auto.sh` and create the folder `resiVecPerPatientSVR_Fold_all` for the test subset. But first, you need to edit some things:
    - `sDirFeatsTest` to point to the folder where you have extracted testing features with the AE 
    - `sDirFeatsTrai` to point to the folder where  there is the training data
    - `ivecDim` : The ivector size you are interested in. 
    - For the number of components, it gets more complicated. You need to write the components that have been selected as the best for at least one per patient tuning. You will get this info there `cat /export/c08/lmorove1/kaldi/egs/beatPDivec/dysk_orig_auto60_400fl/exp/ivec_650/globalAccuPerPatientSVR_Test.log` (it is the dictionary you can see at step 7).

4. Go to `CreateCSV_test.ipynb`. We will use the function `generateCSVtest_per_patient` to create a CSV containing test predictions for all subject_ids. 

7. Provide the variables `best_config`, `dest_dir`, and `src_dir`. The example below are the results of per patient tuning on `dysk_orig_auto60_400fl, ivec: 650`. 

```
best_config = {1004: ['/objs_450_kernel_linear_c_0.002_eps_0.1.pkl', 1.1469489658686098],
 1007: ['/objs_100_kernel_linear_c_0.002_eps_0.1.pkl', 0.09115239389591206],
 1019: ['/objs_400_kernel_linear_c_0.2_eps_0.1.pkl', 0.686931370820251],
 1023: ['/objs_300_kernel_linear_c_0.2_eps_0.1.pkl', 0.8462093717280431],
 1034: ['/objs_100_kernel_linear_c_20.0_eps_0.1.pkl', 0.7961188257851409],
 1038: ['/objs_450_kernel_linear_c_0.002_eps_0.1.pkl', 0.3530848340426855],
 1039: ['/objs_450_kernel_linear_c_0.2_eps_0.1.pkl', 0.3826339325882311],
 1043: ['/objs_300_kernel_linear_c_0.2_eps_0.1.pkl', 0.5525085362997469],
 1044: ['/objs_50_kernel_linear_c_0.002_eps_0.1.pkl', 0.09694768640213237],
 1048: ['/objs_650_kernel_linear_c_0.2_eps_0.1.pkl', 0.4505302952804157],
 1049: ['/objs_250_kernel_linear_c_0.2_eps_0.1.pkl', 0.4001809543831368]}

dest_dir='/export/c08/lmorove1/kaldi/egs/beatPDivec/dysk_orig_auto60_400fl/exp/ivec_650/resiVecSVR_Fold/'
src_dir='/export/c08/lmorove1/kaldi/egs/beatPDivec/dysk_orig_auto60_400fl/exp/ivec_650/resiVecSVR_Fold/'

generateCSVtest_per_patient(src_dir, dest_dir, best_config)
```

The dictionary for best_config is obtained in this file: 
`cat /export/c08/lmorove1/kaldi/egs/beatPDivec/dysk_orig_auto60_400fl/exp/ivec_650/globalAccuPerPatientSVR_Test.log`

8. Run that cell, and it will create a `csv` file in the provided location `dest_dir`. 

**Submission 4**

Again, for this one, we made the mistake of using the best hyperparaneters for Per Patient tuned on `dysk_auto60_400fl_orig` but applied to the wrong features on the test subset : `dysk_auto30_400fl_orig_inactivity_removed`.

We used : 
```
sDirFeatsTest=/export/b19/mpgill/BeatPD/cis_testing_AE_30ft_orig_inactivity_removed
sDirFeatsTrai=/export/b19/mpgill/BeatPD/AE_30ft_orig_inactivity_removed
```
instead of : 
```
sDirFeatsTest=/export/b19/mpgill/BeatPD/cis_testing_AE_60ft_orig
sDirFeatsTrai=/export/b19/mpgill/BeatPD/AE_60ft_400fl_orig
```

<a name="get-preds-trainingtestfolds-perpatient-svr"></a>
#### Option 2 -- For training/test folds 

Just run in `runFor.sh` the script `local/evaluate_global_per_patient_SVR.sh` which will create pkl files needed in `resiVecSVR_Fold*` with the predictions per patient instead of being per configuration. The following excerpt is an example if ivectors files are already created for these dimensions:

```
stage=5 #Features and UBM are already calculated
for  ivecDim in 600 650 700;  do

        sDirRes=`pwd`/exp/ivec_${ivecDim}/
        sDirOut=`pwd`/exp/ivec_${ivecDim}

        local/evaluate_global_per_patient_SVR.sh $sDirRes $sDirOut dysk
done
```

6. Open `CreateFoldsCsv.ipynb`. We will use the function `generateCSVresults_per_patient` to create a CSV containing test predictions for all patients. 

7. Provide the variables `best_config`, `dest_dir`, and `src_dir` like so:

```
best_config = {1004: ['/objs_450_kernel_linear_c_0.002_eps_0.1.pkl', 1.1469489658686098],
 1007: ['/objs_100_kernel_linear_c_0.002_eps_0.1.pkl', 0.09115239389591206],
 1019: ['/objs_400_kernel_linear_c_0.2_eps_0.1.pkl', 0.686931370820251],
 1023: ['/objs_300_kernel_linear_c_0.2_eps_0.1.pkl', 0.8462093717280431],
 1034: ['/objs_100_kernel_linear_c_20.0_eps_0.1.pkl', 0.7961188257851409],
 1038: ['/objs_450_kernel_linear_c_0.002_eps_0.1.pkl', 0.3530848340426855],
 1039: ['/objs_450_kernel_linear_c_0.2_eps_0.1.pkl', 0.3826339325882311],
 1043: ['/objs_300_kernel_linear_c_0.2_eps_0.1.pkl', 0.5525085362997469],
 1044: ['/objs_50_kernel_linear_c_0.002_eps_0.1.pkl', 0.09694768640213237],
 1048: ['/objs_650_kernel_linear_c_0.2_eps_0.1.pkl', 0.4505302952804157],
 1049: ['/objs_250_kernel_linear_c_0.2_eps_0.1.pkl', 0.4001809543831368]}

dest_dir='/export/c08/lmorove1/kaldi/egs/beatPDivec/dysk_orig_auto60_400fl/exp/ivec_650/resiVecPerPatientSVR_Fold_all/'
src_dir='/export/c08/lmorove1/kaldi/egs/beatPDivec/dysk_orig_auto60_400fl/exp/ivec_650/resiVecPerPatientSVR_Fold'

generateCSVresults_per_patient(dest_dir, src_dir, best_config)
```

8. Run that cell, and it will create a `csv` file in the provided location `dest_dir`. The complete path to the file will be printed last : `/export/c08/lmorove1/kaldi/egs/beatPDivec/dysk_noinact_auto30/exp/ivec_650/resiVecPerPatientSVR_Fold_all/preds_per_patient.csv` you will use this file during the fusion with average, in the `sFilePred2` variable. 

<a name="3-tsfresh"></a>
## tsfresh + xgboost  

For this scheme, all the files are in `tsfresh/submit/`. 

```
|-- run.sh : CIS-PD - Submission 3 - run the tsfresh + xgboost scheme without per patient tuning 
|-- run_perpatient.sh : CIS-PD - Submission 4 - run the tsfresh + xgboost scheme with per patient tuning
|-- run_realpd.sh : REAL-PD - Submission 4 - run the tsfresh + xgboost scheme without per patient tuning  
|
|-- conf: ?
|-- data: ?
     |-- label.csv : ??? 
|-- exp: ? 
|-- features: Folder containing the extracted features
     |-- cis-pd.training.csv
     |-- cis-pd.testing.csv 
|
|-- mdl:
     |-- cis-pd.conf : best config for the three subchallenges 
     |-- cis-pd.****.conf : best config tuned per patient for the three subchallenges 
|
|-- src: Folder containing the files to generate features and predictions 
     |
     |--- generator.py: Feature extraction for CIS 
     |
     |--- gridsearch.py: Find best hyperparams and save them to a file
                         (same params for all subjects)
     |
     |--- gridsearch_perpatient.py: Find best hyperparams for each subject
                                    and save them to a file
     |
     |--- predict.py: Predicts and creates submission files
     |--- predict_perpatient.py: Predict with perpatient tuning 
|
|-- submission: Folder containing the CSV files with predictions to submit
|-- submit.sh: ? 
|-- utils: soft link to kaldi/egs/wsj/s5/utils/
```
Prepare the environment and create a symbolic link:

1. Create a softlink from `tsfresh/submit/utils/` to `kaldi/egs/wsj/s5/utils/`. 
2. `cd tsfresh/submit/`
3. `conda create -n BeatPD_xgboost`
4. `source activate BeatPD`
4. `conda install --file requirements_tsfresh_xgboost.txt`

As you can see in our [write-up](https://github.com/Mymoza/BeatPD-CLSP-JHU/wiki/0-Write-Up#final-submission), for the final submission, we used the 4th submission for the three tasks and the two databases, except for CIS-PD and tremor, we decided to go back to our 3rd submission results because that provided us better rankings in the intermediate rounds. 

The following sections explains how to reproduce our final submission. 

### Tremor - Submission 3 for CIS-PD
1. Run `./run.sh`. You might need to make some changes to this file. It is written to be ran on a grid engine. 
    - It will  split the CIS-PD training and testing csv files into 32 subsets and submit 32 jobs to do feature extraction. Then, it will merge all of them to store the features in the `features/` directory. This step only need to be ran once. 
    - Then it will perform a GridSearch, saving the best config 
    - Finally, it will create predictions files to be submitted in the `submission/` folder.  

The same hyperparameters were used for all three tasks so I expect the hyperparameter to generalize. So I did three hyperparameter search on on/off, tremor, dysk and then I compared their performance to see which one is the best. 

For CIS-PD, the best performance was obtained with tremor. 
For REAL-PD, it was watch_gyr tremor. 

For this one, we were not able to reproduce the exact same predictions, we suspect it is because of a random seed. However, the difference in predictions are in the units of 0.001 so it is considered fine. 

🔴 TODO: Compare myself the results in my predictions vs the one 

🔴 TODO: Write how to get kfold_predictions on the kfold and not just on the test subset of the challenge

### Dyskinesia & On/Off - Submission 4 - CIS-PD

The following performs per Patient Tuning.

1. `./run_perpatient.sh`
    - It will perform `gridsearch_perpatient.py` on every task. It will create files in `mdl/cis-pd.on_off.1004.conf`
    - Then, it will create predictions files to be submitted, in the `submission` folder like so : `submission/cis-pd.on_off.perpatient.csv`. 


### Tremor, Dyskinesia & On/Off - Submission 4 - REAL-PD **

The 4th submission of REAL-PD used gridsearch and global normalization.

1. `qsub -l mem_free=30G,ram_free=30G -pe smp 6 -cwd -e /export/b19/mpgill/errors/errors_real_pd_features -o /export/b19/mpgill/outputs/outputs_real_pd_features run_realpd.sh`
    - This will create features in `exp/`, then merge will merge them, like this: `features/watchgyr_total.scp.csv`
    - Then it will perform GridSearch. The same hyperparameters were used for all three tasks so I expect the hyperparameter to generalize. So I did three hyperparameter search on on/off, tremor, dysk and then I compared their performance to see which one is the best. For REAL-PD, it was `watchgyr` and `tremor`. That's why in the code all the other GridSearch combinations are commented out. Only the one used for the 4th submission will be ran. The best hyperparameters found will be stored in `mdl/real-pd.conf`
    - Then we predict the results using `src/predict_realpd.py`. The predictions will be stored in `submission/watchgyr_tremor.csv`. 

**Stop criteria on training data:**

For the 4th submission, we performed early stop with the training data, as that led to some small improvements. To do so, you need to change two lines in the file `src/predict.py`. 

`eval_set=[(tr, tr_y), (te, te_y)]` becomes `eval_set=[(tr, tr_y)]`

`sample_weight_eval_set=[tr_w, te_w]` becomes `sample_weight_eval_set=[tr_w]`.

🦠 Test: without early stop criteria 

🛑TODO: in run_realpd, change the absolute path to our home folder to where labels will be 

🛑TODO: When I reproduce the experiments, do I get the same hyperparameters as Nanxin? 


<a name="4-fusion"></a>
## Fusion

For the second and fourth submission, we performed some fusion of the predictions between an SVR and the xgboost. 

The code to perform the fusion for the fourth submission is in the notebook called `Fusion.ipynb`. 

It is pretty straightforward. Just go to `Dyskinesia - Submission 4 - Average` for an example of how to do fusion evaluation on the test folds. Just give the path to the csv files containing the predictions in `sFilePred1` and `sFilePred2` (obtained [here](#get-preds-trainingtestfolds-perpatient-svr)), like so:

```
sFilePred1='/home/mpgill/BeatPD/BeatPD-CLSP-JHU/tsfresh/submit/submission4_preds/kfold_prediction_dyskinesia.csv'
sFilePred2='/export/b19/mpgill/kaldi/egs/beatPDivec/dysk_orig_auto60_400fl_scratch/exp/ivec_650/resiVecSVR_Fold/preds_per_patient.csv'
```

You will get results: 

```
--- MSEscore ---
Final score :  0.4830357155225596
Overall MSE Classif. 1 - tsfresh:  None
--- MSEscore ---
Final score :  0.5144468970875267
Overall MSE Classif. 2 - ivec:  None
--- MSEscore ---
Final score :  0.48601343286255055
Overall MSE Fusion - average :  None
```


# References 

- Challenge website https://www.synapse.org/#!Synapse:syn20825169/wiki/596118 
- tsfresh github 
- ivectors paper? 

🛑TODO: Check that all links to the wiki are still valid 
