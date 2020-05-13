# BEATPD 

This GitHub repository contains the code to reproduce the results obtained by the team JHU-CLSP during the BeatPD challenge. The challenge had 4 submission rounds before the final submission (hereafter addressed as 1<sup>st</sup> submission, 2<sup>nd</sup> submission, 3<sup>rd</sup> submission, 4<sup>th</sup> submission, final submission)
<br>

For the final submission, we submitted:
- `ON/OFF`:
    - CIS-PD: same as 4th submission
    - REAL-PD: same as 4th submission
- `Tremor`:
    - CIS-PD: same as 3rd submission
    - REAL-PD: same as 4th submission
- `Dyskinesia`:
    - CIS-PD: same as 4th submission
    - REAL-PD: same as 4th submission
    
<br>

This README walks you through re-creating our final submission. If you would like to re-create all submissions, please follow our [wiki documentation](https://github.com/Mymoza/BeatPD-CLSP-JHU/wiki/0-Write-Up). The wiki also contains detailed explanation of all our approaches.
<br>

# Step-By-Step guide 

This step-by-step guide will cover the following steps: 


- [Data Pre-Processing](#1-data-pre-processing)
- Approach I : [TSFRESH + XGBOOST](#3-tsfresh)
- Approach II :  [AutoEncoder (AE)](#2.2.2-get-ae-features) + [i-vectors](#2.3-create-i-vectors) + [SVRs](#2.4-get-results)
- Approach III : [Fusion](#4-fusion)

<hr>

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

<hr>

<a name="1-data-pre-processing"></a>
## Data Pre-Processing 

All the steps to prepare the data is done in the Jupyter Notebook `prepare_data.ipynb`. 

1. Open the notebook
2. Change the `data_dir` variable for the absolute path to the folder that contains the data given by the challenge. In this folder, you should already have the following directories downloaded from the [challenge website](https://www.synapse.org/#!Synapse:syn20825169/wiki/596118): 
```
/export/b19/mpgill/BeatPD_data  $ ls
cis-pd.ancillary_data  cis-pd.testing_data   real-pd.ancillary_data  real-pd.testing_data
cis-pd.data_labels     cis-pd.training_data  real-pd.data_labels     real-pd.training_data
```
3. Execute the cells in the Notebook. It will create several folders needed to reproduce the experiments. The [data directory structure is documented in the wiki](https://github.com/Mymoza/BeatPD-CLSP-JHU/wiki/1-Data-Directory-Structure).

<hr>

<a name="3-tsfresh"></a>
##  Approach I : tsfresh + xgboost  

For this scheme, all the files are in `<your-path-to-AE-features>/tsfresh/submit/`. 

```
|-- run.sh : CIS-PD - Submission 3 - run the tsfresh + xgboost scheme without per patient tuning 
|-- run_perpatient.sh : CIS-PD - Submission 4 - run the tsfresh + xgboost scheme with per patient tuning
|-- run_realpd.sh : REAL-PD - Submission 4 - run the tsfresh + xgboost scheme without per patient tuning  
|
|-- conf: 
|-- data: Challenge data
     |-- label.csv  
|-- exp: Feature extraction jobs that were divided in 32 subsets
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
|-- utils: soft link to <your_path_to_kaldi>/kaldi/egs/wsj/s5/utils/
```
Prepare the environment and create a symbolic link:

1. Create a softlink to `<your_path_to_kaldi>/kaldi/egs/wsj/s5/utils/` at `<your-path-to-AE-features>/tsfresh/submit/utils/` using `ln -s <your_path_to_kaldi>/kaldi/egs/wsj/s5/utils/ to <your-path-to-AE-features>/tsfresh/submit/utils/`
2. `cd <your-path-to-AE-features>/tsfresh/submit/`
3. `conda create -n BeatPD_xgboost`
4. `source activate BeatPD_xgboost`
4. `conda install --file requirements_tsfresh_xgboost.txt`

As you can see in our [write-up](https://github.com/Mymoza/BeatPD-CLSP-JHU/wiki/0-Write-Up#final-submission), for the final submission, we used the 4<sup>th </sup> submission for the three tasks and the two databases, except for CIS-PD and tremor, we decided to go back to our 3rd submission results because that provided us better rankings in the intermediate rounds. 

The following sections explains how to reproduce our final submission. 

### Tremor - Submission 3 for CIS-PD
1. In `run.sh`, in the section to generate submission files, edit the absolute path to the `CIS-PD_Test_Data_IDs_Labels.csv` that is currently hardcoded. 
2. Run `./run.sh`. You might need to make some changes to this file. It is written to be ran on a grid engine. 
    - It will  split the CIS-PD training and testing csv files into 32 subsets and submit 32 jobs to do feature extraction. Then, it will merge all of them to store the features in the `features/` directory. This step only need to be ran once. 
    - Then it will perform a GridSearch, saving the best config 
    - Finally, it will create predictions files to be submitted in the `submission/` folder.  

The same hyperparameters were used for all three tasks so I expect the hyperparameter to generalize. So I did three hyperparameter search on on/off, tremor, dysk and then I compared their performance to see which one is the best. 

For CIS-PD, the best performance was obtained with tremor. 
For REAL-PD, it was watch_gyr tremor. 

For this one, we were not able to reproduce the exact same predictions, we suspect it is because of a random seed. However, the difference in predictions are in the units of 0.001 so it is considered fine. 

### Dyskinesia & On/Off - Submission 4 - CIS-PD

The following performs per Patient Tuning.

1. In `run_perpatient.sh`, in the section to generate submission files, edit the absolute path to the `CIS-PD_Test_Data_IDs_Labels.csv` that is currently hardcoded. 
2. `./run_perpatient.sh`
    - It will perform `gridsearch_perpatient.py` on every task. It will create files in `mdl/cis-pd.on_off.1004.conf`
    - Then, it will create predictions files to be submitted, in the `submission` folder like so : `submission/cis-pd.on_off.perpatient.csv`. 


### Tremor, Dyskinesia & On/Off - Submission 4 - REAL-PD **

The 4th submission of REAL-PD used gridsearch and global normalization.

1. In `run_realpd.sh`, edit the absolute path hardcoded to the REAL-PD labels and write your own path to the labels you downloaded from the website of the challenge. 
2. Run `./run_realpd.sh`
    - This will create features in `exp/`, then merge will merge them, like this: `features/watchgyr_total.scp.csv`
    - Then it will perform GridSearch. The same hyperparameters were used for all three tasks so I expect the hyperparameter to generalize. So I did three hyperparameter search on on/off, tremor, dysk and then I compared their performance to see which one is the best. For REAL-PD, it was `watchgyr` and `tremor`. That's why in the code all the other GridSearch combinations are commented out. Only the one used for the 4th submission will be ran. The best hyperparameters found will be stored in `mdl/real-pd.conf`
    - Then we predict the results using `src/predict_realpd.py`. The predictions will be stored in `submission/watchgyr_tremor.csv`. 

**Stop criteria on training data:**

For the 4th submission, we performed early stop with the training data, as that led to some small improvements. To do so, you need to change two lines in the file `src/predict.py`. 

`eval_set=[(tr, tr_y), (te, te_y)]` becomes `eval_set=[(tr, tr_y)]`

`sample_weight_eval_set=[tr_w, te_w]` becomes `sample_weight_eval_set=[tr_w]`.

🛑TODO: in run_realpd, change the absolute path to our home folder to where labels will be 


<hr>

##  Approach II

<a name="2-embeddings"></a>
### AutoEncoder (AE) features 

<a name="2.2.1-train-ae"></a>
#### Train the AutoEncoder 

1. At the moment, all the code needed for the AE lives [on a branch](https://github.com/Mymoza/BeatPD-CLSP-JHU/pull/14). So the first step is to checkout that branch with `git checkout marie_ml_dl_real`.
2. `conda env create --file environment_ae.yml` : This will create the `keras_tf2` environment you need to run AE experiments.
3. Train an AE model & save their features:
    - For CIS-PD: At line 51 of the `train_AE.py` file, change the `save_dir` path to the directory where you want to store the AE models. 
    - For REAL-PD: At line 53 of the `train_AE_real.py` file, change the `save_dir` path to the directory where you want to store the AE models.
4. Launch the training for the configurations you want. Some examples are available in this wiki page about [Creating AutoEncoder Features](https://github.com/Mymoza/BeatPD-CLSP-JHU/wiki/2-Creating-AutoEncoder-Features). To reproduce the results of submission 4, you will need the following command which uses features of length 60 and 400 as frame length: 

`python train_AE.py --latent_dim 60 -dlP '{"remove_inactivity":"False"}' --saveAEFeats --saveFeatDir "/export/b19/mpgill/BeatPD/AE_60ft_400fl_orig/"`

5. This should create the following file `mlp_encoder_uad_False_ld_60.h5` and the features will be saved in the directory provided with the `--saveFeatDir` flag. 

<a name="2.2.2-get-ae-features"></a>
#### Get AutoEncoder (AE) Features 

1. `git checkout marie_ml_dl_real`: The code to get features from the AutoEncoder is in another branch. 
2. `cd ml_dl`
3. `source activate keras_tf2`
4. Go to this [wiki page](https://github.com/Mymoza/BeatPD-CLSP-JHU/wiki/2-Creating-AutoEncoder-Features) that lists many examples of commands you can use the create the required AE features. If you only want to get features without creating models, you need to comment a section of the `train_AE.py` and `train_AE_real.py` files. The section needed to be commented is identified directly in the file.
5. Run the command you are interested in getting!

**Submission 4**
To reproduce it, you will need to create the following:
- `AE_30ft_orig_inactivity_removed`
- `cis_testing_AE_30ft_orig_inactivity_removed`

We made a mistake and although we meant to be using `AE_60ft_400fl_orig` (and `cis_testing_AE_60ft_orig`) as that provided us with better crossvalidation results, we ended up using the wrong features! 

<a name="2.3-create-i-vectors"></a>
### Create i-vectors 
 
After creating Autoencoder features or the MFCC, we can create i-vectors. 

You need to install [Kaldi](https://kaldi-asr.org) using either [official install instructions](https://kaldi-asr.org/doc/install.html) or [easy install instructions](http://jrmeyer.github.io/asr/2016/01/26/Installing-Kaldi.html) if you find the official one difficult to follow.

The following steps will vary a lot depending on what i-vector you want to create. You will need to create `dysk_orig_auto60_400fl` for the 4th submission.

🛑TODO: Good vocabulary? 

1. `cd <your-path-to-kaldi>/kaldi/egs/` : Change your directory to where you installed Kaldi. 
2. `mkdir beatPDivec; cd beatPDivec` : Create a directory to hold the i-vectors. 
3.  `cp <your-path-github-repo>/sid_novad/* ../sre08/v1/sid/.` : Copy the `novad.sh` files from the repository to your Kaldi's directory 
4. `mkdir *****` : Create a folder with a meaningful name about the i-vectors we want to create. The nomenclature we used to name the i-vectors we created was also [documented in the wiki](https://github.com/Mymoza/BeatPD-CLSP-JHU/wiki/4-i-vectors-nomenclature). To reproduce the final submission, create `dysk_orig_auto60_400fl`.
5. `cd ****` : Change your directory to the i-vector folder you just created 
6. `mkdir data`
7. `cp -rf {path-github-repo}/beatPDivec/default_data/v2_auto/. .`
8. `cp -rf {path-github-repo}/beatPDivec/default_data/autoencData/data/{onoff - tremor - dyskinesia}/. data/.` : Copy the data for the task. In this case, we used dyskinesia. 
9. `ln -s sid ../../sre08/v1/sid; ln -s steps ../../sre08/v1/steps; ln -s utils ../../sre08/v1/utils` : Create symbolic links
10. `vim runFor.sh`: Edit the following variables:
    - `subChallenge`: use either `onoff`, `tremor`, or `dysk`. 
    - `sDirFeats`: use the absolute path to the AE features you want to use, for example `sDirFeats={path-to-AE-features}/AE_30ft_orig_inactivity_removed` 
11. `qsub -l mem_free=30G,ram_free=30G -pe smp 6 -cwd -e errors/errors_dysk_noinact_auto30 -o outputs/outputs_dysk_noinact_auto30 runFor.sh`

🔴TODO : remove qsub instructions? 

<a name="2.4-get-results"></a>
### Get results on test folds for SVR

The file `runFor.sh` will create the log files with the results of the experiments you ran. The following section explains how to retrieve those results. 
#### Manually - for one size of i-vector 
The following example will retrieve results for the following i-vector: `trem_noinact_auto30`.

1. `cd <your-path-to-kaldi>/kaldi/egs/beatPDivec/trem_noinact_auto30/exp/`
2. `cd ivec_350` : Then, choose an i-vector size 
3. `ls` : 
```
globalAccuPLDA.log : Result for PLDA 
globalAccuKNN.log : Result for KNN
globalAccuSVR_Test.log : Result for SVR 
globalAccuPerPatientSVR_Test.log : Result for Per Patient SVR 
globalAccuEveryoneSVR_Test.log : Result for Everyone SVR
```

#### Extract results for different i-vector sizes 

As of now, the automation is present in the `get_excel_results.ipynb`, and just creates a table in Jupyter from which we can copy and paste to Excel or Google spreadsheet:

So far, it was only developed for Per Patient SVR and Everyone SVR results.
For the other back-ends, you still need to get the results by hand like it was explained in the previous section. 

<a name="2.5-get-predictions"></a>
### Get Predictions CSV 

### Per Patient SVR 

#### Option 1 - For the test subset of the challenge 

1. `cd` to the i-vector location, for example `cd <your-path-to-kaldi>/kaldi/egs/beatPDivec/dysk_orig_auto60_400fl/` was the i-vector used for the [4th submission](https://github.com/Mymoza/BeatPD-CLSP-JHU/wiki/0-Write-Up).
2. In the file `<your-path-to-github-repo>/beatPDivec/default_data/v2_auto/local/pca_svr_bpd2.sh`, make sure that the flag `--bPatientPredictionsPkl` is added to create pkl files for each subject_id, like this:

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
    - `ivecDim` : The i-vector size you are interested in. 
    - For the number of components, it gets more complicated. You need to write the components that have been selected as the best for at least one per patient tuning. You will get this info there `cat <your-path-to-kaldi>/kaldi/egs/beatPDivec/dysk_orig_auto60_400fl/exp/ivec_650/globalAccuPerPatientSVR_Test.log` (it is the dictionary you can see at step 7).

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

dest_dir='<your-path-to-kaldi>/kaldi/egs/beatPDivec/dysk_orig_auto60_400fl/exp/ivec_650/resiVecSVR_Fold/'
src_dir='<your-path-to-kaldi>/kaldi/egs/beatPDivec/dysk_orig_auto60_400fl/exp/ivec_650/resiVecSVR_Fold/'

generateCSVtest_per_patient(src_dir, dest_dir, best_config)
```

The dictionary for best_config is obtained in this file: 
`cat <your-path-to-kaldi>/kaldi/egs/beatPDivec/dysk_orig_auto60_400fl/exp/ivec_650/globalAccuPerPatientSVR_Test.log`

8. Run that cell, and it will create a `csv` file in the provided location `dest_dir`. 

**Submission 4**

For this submission, we made the mistake of using the best hyperparameters for Per Patient tuned on `dysk_auto60_400fl_orig` but applied to the wrong features on the test subset : `dysk_auto30_400fl_orig_inactivity_removed`. But similar cross-validation results were obtained on both these i-vectors.

We used : 
```
sDirFeatsTest=<your-path-to-AE-features>/cis_testing_AE_30ft_orig_inactivity_removed
sDirFeatsTrai=<your-path-to-AE-features>/AE_30ft_orig_inactivity_removed
```
instead of : 
```
sDirFeatsTest=<your-path-to-AE-features>/cis_testing_AE_60ft_orig
sDirFeatsTrai=<your-path-to-AE-features>/AE_60ft_400fl_orig
```

<a name="get-preds-trainingtestfolds-perpatient-svr"></a>
#### Option 2 -- For training/test folds 

Run `runFor.sh`, which will call the script `<your-path-to-github-repo>/beatPDivec/default_data/v2_auto/local/evaluate_global_per_patient_SVR.sh` that will create pkl files needed in `resiVecSVR_Fold*` with the predictions per patient instead of being per configuration. The following excerpt is an example if i-vectors files are already created for these dimensions:

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

dest_dir='<your-path-to-kaldi>/kaldi/egs/beatPDivec/dysk_orig_auto60_400fl/exp/ivec_650/resiVecPerPatientSVR_Fold_all/'
src_dir='<your-path-to-kaldi>/kaldi/egs/beatPDivec/dysk_orig_auto60_400fl/exp/ivec_650/resiVecPerPatientSVR_Fold'

generateCSVresults_per_patient(dest_dir, src_dir, best_config)
```

8. Run that cell, and it will create a `csv` file in the provided location `dest_dir`. The complete path to the file will be printed last : `<your-path-to-kaldi>/kaldi/egs/beatPDivec/dysk_noinact_auto30/exp/ivec_650/resiVecPerPatientSVR_Fold_all/preds_per_patient.csv` you will use this file during the fusion with average, in the `sFilePred2` variable. 






<hr>

<a name="4-fusion"></a>
## Approach III : Fusion

For the second and fourth submission, we performed some fusion of the predictions between an SVR and the xgboost. 

The code to perform the fusion for the fourth submission is in the notebook called `Fusion.ipynb`. 

It is pretty straightforward. Just go to `Dyskinesia - Submission 4 - Average` for an example of how to do fusion evaluation on the test folds. Just give the path to the csv files containing the predictions in `sFilePred1` and `sFilePred2` (obtained [here](#get-preds-trainingtestfolds-perpatient-svr)), like so:

```
sFilePred1='<your-path-to-github-repo>/BeatPD-CLSP-JHU/tsfresh/submit/submission4_preds/kfold_prediction_dyskinesia.csv'
sFilePred2='<your-path-to-kaldi>/kaldi/egs/beatPDivec/dysk_orig_auto60_400fl_scratch/exp/ivec_650/resiVecSVR_Fold/preds_per_patient.csv'
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

<hr>


# References 

- [The Biomarker and Endpoint Assessment to Track Parkinson's Disease (BEAT-PD) Challenge](https://www.synapse.org/#!Synapse:syn20825169/wiki/596118)
- Christ, M., Braun, N., Neuffer, J. and Kempa-Liehr A.W. (2018). Time Series FeatuRe Extraction on basis of Scalable Hypothesis tests (tsfresh -- A Python package). Neurocomputing 307 (2018) 72-77, doi:10.1016/j.neucom.2018.03.067. [GitHub](https://github.com/blue-yonder/tsfresh)
- Dehak, Najim, et al. "Front-end factor analysis for speaker verification." IEEE Transactions on Audio, Speech, and Language Processing 19.4 (2010): 788-798. 
