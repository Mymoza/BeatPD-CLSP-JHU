# BEATPD 



This GitHub repository contains the code to reproduce the results obtained by the team JHU-CLSP during the [BeatPD challenge](https://www.synapse.org/#!Synapse:syn20825169/wiki/596118).
Data description and copyright: 
<blockquote>
These data were generated by participants of The Michael J. Fox Foundation for Parkinson's Research Mobile or Wearable Studies. They were obtained as part of the Biomarker & Endpoint Assessment to Track Parkinson's Disease DREAM Challenge (through Synapse ID syn20825169) made possible through partnership of The Michael J. Fox Foundation for Parkinson's Research, Sage Bionetworks, and BRAIN Commons
</blockquote>

The challenge had 4 submission rounds before the final submission. Hereafter, they are addressed as 1st submission, 2nd submission, 3rd submission, 4th submission, final submission.
<br>

For the final submission, we submitted:
- `ON/OFF`:
    - CIS-PD: same as 3rd submission + foldaverage
    - REAL-PD: same as 3rd submission + foldaverage 
- `Tremor`:
    - CIS-PD: same as 3rd submission
    - REAL-PD: same as 3rd submission
- `Dyskinesia`:
    - CIS-PD: same as 4th submission
    - REAL-PD: same as 3rd submission
    
<br>

This README walks you through re-creating our final submission. For detailed write-up and to re-create all submissions, please follow our [wiki documentation](https://github.com/Mymoza/BeatPD-CLSP-JHU/wiki/0-Write-Up). 

<br>

## Approaches
We have followed 3 approaches during the course of all our submissions :

![Summary of the three approaches followed](https://github.com/Mymoza/BeatPD-CLSP-JHU/blob/master/images/general-pipeline.png )


Please note that due to a lack of development dataset, for all approaches, we performed 5-fold cross-validation and analyzed results of each dataset (CIS-PD and Real-PD) separately.

- <b> Approach I : TSFRESH + XGBOOST </b> 

tsfresh extracts statistical data features from the signal and xgboost handles tabular data extraction and uses decision trees to select important features and combine them to make strong predictions.

- <b> Approach II :  AE + i-vector + Support Vector Regression (SVR) </b> 

The problem with using Deep Neural Network-based techniques directly on signals from wearble devices is that there is only one label for a 20 minute file. So, the first step is to reduce the raw signal to features. We used an DNN based auto-encoder (AE) to extract features. Later we use respresentatial learning method called `i-vector` to convert the features into a vector of fixed size, regardless of the length of the signal. In this way, we used a combination of trained AE and i-vector extractor to obtain a single (fixed sized) vector per signal. Using i-vectors as features, we used Support Vector Regression (SVR) with linear kernel to predict the labels.

- <b>  Approach III : Fusion </b> 

A fusion of the predictions from Approach I and Approach II was done using either: 
-  Gradient boosting regression. The regressor was trained with the predicted labels from the testing folds from cross-validations.
- Average of predictions


<br>
<hr>
<br>

# Step-By-Step guide for setting up environment and data

This step-by-step guide will cover the following steps: 
- [Clone our repository from git](#git-clone)
- [Set up the environment](#0-set-up-env)
- [Data Pre-Processing](#1-data-pre-processing)
- [Code for all approaches](#2-code)
    - Approach I : [TSFRESH + XGBOOST](#3-tsfresh)
    - Approach II :  [AutoEncoder (AE)](#2.2.2-get-ae-features) + [i-vectors](#2.3-create-i-vectors) + [SVRs](#2.4-get-results)
    - Approach III : [Fusion](#4-fusion)


<a name="git-clone"></a>
## Clone our repository from git :

In your terminal, run the following command to clone our repository from git
```
$ git clone https://github.com/Mymoza/BeatPD-CLSP-JHU.git
```

<a name="0-set-up-env"></a>
## Set up the environment : 
We use python for majority of our scripts. We use jupyter notebook to facilitate an interactive envirnment. To run our scripts, please create an environment using `requirements.txt` file by following these steps:

```
$ conda create -n BeatPD python=3.5
$ source activate BeatPD 
$ conda install --file requirements.txt
```

* <b>Note:</b> Make sure that the Jupyter notebook is running on `BeatPD` kernel. 

If the conda environment isn't showing in Jupyter kernels (Kernel > Change Kernel > BeatPD), run: 
```
$ ipython kernel install --user --name=BeatPD
```
You will then be able to select `BeatPD` as your kernel. 

## Install kaldi :

You need to install [Kaldi](https://kaldi-asr.org). For installation, you can use either the [official install instructions](https://kaldi-asr.org/doc/install.html) or  the [easy install instructions](http://jrmeyer.github.io/asr/2016/01/26/Installing-Kaldi.html) if you find the official one difficult to follow.

<br>
<hr>
<br>

<a name="1-data-pre-processing"></a>
## Data Pre-Processing 

First step is to prepare the data given by the challenge. 

1. Download the training_data, the ancillary_data and the testing_data from the [challenge website](https://www.synapse.org/#!Synapse:syn20825169/wiki/600903)
2. `mkdir BeatPD_data` Create a folder to contain all the files `.tar.bz2` you just downloaded for the challenge 
3. `tar -xvf cis-pd.data_labels.tar.bz2; mv data_labels cis-pd.data_labels` it will extract a folder that we will rename to make it clear that it contains the label for the CIS-PD database 
4. `tar -xvf real-pd.data_labels.tar.bz2; mv data_labels real-pd.data_labels`: same thing but for the REAL-PD database
5. `rm -rf *.tar.bz2` : remove the compressed folders now that we have extracted the data we need. 
6. `tar -xvf real-pd.training_data_updated.tar.bz2; mv training_data/ real-pd.training_data; rm  real-pd.training_data_updated.tar.bz2; 
7. `tar -xvf cis-pd.training_data.tar.bz2; mv training_data/ cis-pd.training_data; rm cis-pd.training_data.tar.bz2;
8. `tar -xvf cis-pd.ancillary_data.tar.bz2; mv ancillary_data/ cis-pd.ancillary_data; rm cis-pd.ancillary_data.tar.bz2;` 
9. `tar -xvf real-pd.ancillary_data_updated.tar.bz2; mv ancillary_data real-pd.ancillary_data; rm real-pd.ancillary_data_updated.tar.bz2;` 
10. `tar -xvf cis-pd.testing_data.tar.bz2; mv testing_data/ cis-pd.testing_data/; rm  cis-pd.testing_data.tar.bz2` 
11. `tar -xvf real-pd.testing_data_updated.tar.bz2; mv testing_data/ real-pd.testing_data/; rm real-pd.testing_data_updated.tar.bz2;` 
12. `mv cis-pd.CIS-PD_Test_Data_IDs.csv CIS-PD_Test_Data_IDs_Labels.csv; mv CIS-PD_Test_Data_IDs_Labels.csv cis-pd.data_labels/;` The labels for the test subset comes in just a `csv` file, so put that file in the `data_labels` folder. 
13. `mv real-pd.REAL-PD_Test_Data_IDs.csv REAL-PD_Test_Data_IDs_Labels.csv; mv REAL-PD_Test_Data_IDs_Labels.csv real-pd.data_labels/` 

All the steps to do pre-processing on the data is done in the Jupyter Notebook `prepare_data.ipynb`. 

1. Open the notebook
2. Change the `data_dir` variable for the absolute path to the folder that contains the data given by the challenge. In this folder, you should already have the following directories downloaded from the [challenge website](https://www.synapse.org/#!Synapse:syn20825169/wiki/596118): 
```
<path-to-BeatPD_data>  $ ls
cis-pd.ancillary_data  cis-pd.testing_data   real-pd.ancillary_data  real-pd.testing_data
cis-pd.data_labels     cis-pd.training_data  real-pd.data_labels     real-pd.training_data
```
3. Execute the cells in the Notebook. It will create several folders needed to reproduce the experiments. The [data directory structure is documented in the wiki](https://github.com/Mymoza/BeatPD-CLSP-JHU/wiki/1-Data-Directory-Structure).


<br>
<hr>
<br>

<a name="2-code"></a>
## Code for all approaches

<a name="3-tsfresh"></a>
##  Approach I : tsfresh + xgboost  

For this scheme, all the files are in `<path-github-repo>/tsfresh/submit/`. 

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
|-- utils: soft link to kaldi/egs/wsj/s5/utils/
```
Prepare the environment and create a symbolic link:

1. Create a softlink from `tsfresh/submit/utils/` to `kaldi/egs/wsj/s5/utils/`. 
2. `cd tsfresh/submit/`
3. `conda create -n BeatPD_xgboost`
4. `source activate BeatPD_xgboost`
5. `conda install --file requirements_tsfresh_xgboost.txt`
6. In the data/ folder, add `BEAT-PD_SC1_OnOff_Submission_Template.csv`, `BEAT-PD_SC2_Dyskinesia_Submission_Template.csv` and `BEAT-PD_SC3_Tremor_Submission_Template.csv` downloaded from the challenge 

As you can see in our [write-up](https://github.com/Mymoza/BeatPD-CLSP-JHU/wiki/0-Write-Up#final-submission), for the final submission, the following sections need to be generated to create predictions files for tsfresh.  

The following sections explains how to reproduce our final submission. 

### ON/OFF - Submission 5 (final submission) for CIS-PD and REAL-PD

Instead of training one model on whole training set, we used our 5-fold to get five different models. We averaged predictions from those five models. The benefit of this approach is that for each model, we can use the test fold to do the early stop to avoid overfitting. Also combination of five systems may improve the overall performance.

1. In `run_foldaverage.sh`, edit the absolute path to the `CIS-PD_Test_Data_IDs_Labels.csv` and `REAL-PD_Test_Data_IDs_Labels.csv` that are currently hardcoded. 
2. Run `run_foldaverage.sh`, which will run the necessary code for both databases. It will create the following files:
    1. `submission/cis-pd.on_off_new.csv` files containing predictions on the test subset for CIS-PD.
    2. `submission/<watchgyr - watchacc - phoneacc>_on_off.csv` : For REAL-PD on test subset 

### Tremor - Submission 3 for CIS-PD & REAL-PD

1. In `run.sh`, in the section to generate submission files, edit the absolute path to the `CIS-PD_Test_Data_IDs_Labels.csv` that is currently hardcoded. 
2. Run `./run.sh`. You might need to make some changes to this file. It is written to be ran on a grid engine. 
    - It will  split the CIS-PD training and testing csv files into 32 subsets and submit 32 jobs to do feature extraction. Then, it will merge all of them to store the features in the `features/` directory. This step only need to be ran once. 
    - Then it will perform a GridSearch, saving the best config 
    - Finally, it will create predictions files to be submitted in the `submission/` folder.  

The same hyperparameters were used for all three tasks so I expect the hyperparameter to generalize. So I did three hyperparameter search on on/off, tremor, dysk and then I compared their performance to see which one is the best. 

For CIS-PD, the best performance was obtained with tremor. 
For REAL-PD, it was watch_gyr tremor. 

For this one, we were not able to reproduce the exact same predictions, we suspect it is because of a random seed. However, the difference in predictions are in the units of 0.001 so it is considered fine. 

### Dyskinesia - CIS-PD & REAL-PD
 
#### Submission 4 - CIS-PD

The following performs per Patient Tuning which we submitted in the 4th intermediate round. The following is for the CIS-PD database.

1. In `run_perpatient.sh`, in the section to generate submission files, edit the absolute path to the `CIS-PD_Test_Data_IDs_Labels.csv` that is currently hardcoded. 
2. `./run_perpatient.sh`
    - It will perform `gridsearch_perpatient.py` on every task. It will create files in `mdl/cis-pd.on_off.1004.conf`
    - Then, it will create predictions files to be submitted, in the `submission` folder like so : `submission/cis-pd.on_off.perpatient.csv`. 


#### Submission 3 - REAL-PD

1. In `run_realpd.sh`, edit the absolute path hardcoded to the REAL-PD labels and write your own path to the labels you downloaded from the website of the challenge. 
2. Run `./run_realpd.sh`
    - This will create features in `exp/`, then merge will merge them, like this: `features/watchgyr_total.scp.csv`
    - Then it will perform GridSearch. The same hyperparameters were used for all three tasks so I expect the hyperparameter to generalize. So I did three hyperparameter search on on/off, tremor, dysk and then I compared their performance to see which one is the best. For REAL-PD, it was `watchgyr` and `tremor`. That's why in the code all the other GridSearch combinations are commented out. Only the one used for the 4th submission will be ran. The best hyperparameters found will be stored in `mdl/real-pd.conf`
    - Then we predict the results using `src/predict_realpd.py`. The predictions will be stored in `submission/watchgyr_tremor.csv`. 

**Stop criteria on training data:**

For the 4th submission, we performed early stop with the training data, as that led to some small improvements. To do so, you need to change two lines in the file `src/predict.py`. 

`eval_set=[(tr, tr_y), (te, te_y)]` becomes `eval_set=[(tr, tr_y)]`

`sample_weight_eval_set=[tr_w, te_w]` becomes `sample_weight_eval_set=[tr_w]`.

<!---
 🛑TODO: in run_realpd, change the absolute path to our home folder to where labels will be
-->

<hr>

##  Approach II : AE + i-vectors + SVR

For dyskinesia, in the final submission, we performed a fusion with the average of the predictions between Approach 1 and Approach 2. The following section will help you create the files needed to perform the fusion. 

<a name="2-embeddings"></a>
### AutoEncoder (AE) features 

<a name="2.2.1-train-ae"></a>
#### Train the AutoEncoder 

1. At the moment, all the code needed for the AE lives [on a branch](https://github.com/Mymoza/BeatPD-CLSP-JHU/pull/14). So the first step is to checkout that branch with `git checkout marie_ml_dl_real`.
2. `conda env create --file environment_ae.yml` : This will create the `keras_tf2` environment you need to run AE experiments.
3. Train an AE model & save their features:
    - For CIS-PD: At line 51 of the `train_AE.py` file, change the `save_dir` path to the directory where you want to store the AE models, which will be referred to as `<your-path-to-AE-Features>`. 
    - For REAL-PD: At line 53 of the `train_AE_real.py` file, change the `save_dir` path to the directory where you want to store the AE models.
4. Launch the training for the configurations you want. Some examples are available in this wiki page about [Creating AutoEncoder Features](https://github.com/Mymoza/BeatPD-CLSP-JHU/wiki/2-Creating-AutoEncoder-Features). To reproduce the results of submission 4, you will need the following command which uses features of length 30 and a framelength of 400, with the inactivty removed: 

```
python train_AE.py --saveAEFeats -dlP '{"remove_inactivity": "True", "my_data_path": "<path-to-BeatPD-data>/cis-pd.training_data/", "my_mask_path": "<your-path-to-AE-features>/cis-pd.training_data.high_pass_mask/"}' --saveFeatDir "<your-path-to-AE-features>/AE_30ft_orig_inactivity_removed/"
```

5. This should create the following file `<your-path-to-AE-features>/<Weights>/mlp_encoder_uad_False_ld_30.h5` and the features will be saved in the directory provided with the `--saveFeatDir` flag. 

6. Also generate features on the testing subset of the challenge with the following command: 

```
python test_AE.py -dlP '{"my_data_path": "<path-to-BeatPD-data>/cis-pd.testing_data/", "my_mask_path": "<your-path-to-AE-features>/cis-pd.testing_data.high_pass_mask/", "remove_inactivity": "True"}' --saveAEFeats --saveFeatDir "<your-path-to-AE-features>/cis_testing_AE_30ft_orig_inactivity_removed"
``` 

<!---
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
-->

<a name="2.3-create-i-vectors"></a>
### Create i-vectors 
 
After creating Autoencoder features, we can create i-vectors. The following steps will vary a lot depending on what i-vector you want to create. You will need to create `dysk_noinact_auto30` to reproduce our final submission.

1. `cd <your-path-to-kaldi>/kaldi/egs/` : Change your directory to where you installed Kaldi. 
2. `mkdir beatPDivec; cd beatPDivec` : Create a directory to hold the i-vectors. 
3.  `cp <your-path-github-repo>/sid_novad/* ../sre08/v1/sid/.` : Copy the `novad.sh` files from the repository to your Kaldi's directory 
4. `mkdir <i-vector-name>` : Create a folder with a meaningful name about the i-vectors we want to create. The nomenclature we used to name the i-vectors we created was also [documented in the wiki](https://github.com/Mymoza/BeatPD-CLSP-JHU/wiki/4-i-vectors-nomenclature). To reproduce the final submission, create `dysk_noinact_auto30`.
5. `cd <i-vector-name>` : Change your directory to the i-vector folder you just created 
6. `mkdir data`
7. `cp -rf <your-path-github-repo>/beatPDivec/default_data/v2_auto/. .`
8. `cp -rf <your-path-github-repo>/beatPDivec/default_data/autoencData/data/<onoff - tremor - dyskinesia>/. data/.` : Copy the data for the task. For the final submission, use `dyskinesia`. 
9. `ln -s sid ../../sre08/v1/sid; ln -s steps ../../sre08/v1/steps; ln -s utils ../../sre08/v1/utils` : Create symbolic links
10. `vim runFor.sh`: Edit the following variables:
    - `subChallenge`: use either `onoff`, `tremor`, or `dysk`. 
    - `sDirFeats`: use the absolute path to the AE features you want to use. For the final submission, use `sDirFeats=<path-to-AE-features>/AE_30ft_orig_inactivity_removed`
11. `./runFor.sh`

<a name="2.4-get-results"></a>
### Get results on test folds for SVR

This section is only used to get cross-validation results. You can skip this section and just <a href="#2.5-get-predictions">get a CSV file with predictions</a> right away.

<details>
  <summary>Expand: get cross-validation results on test folds</summary>
  <div>The file `runFor.sh` will create the log files with the results of the experiments you ran. The following section explains how to retrieve those results.

#### Manually - for one size of i-vector
The following example will retrieve results for the following i-vector: `trem_noinact_auto30`.

1. `cd <your-path-to-kaldi>/kaldi/egs/beatPDivec/dysk_noinact_auto30/exp/`
2. `cd ivec_650` : Then, choose an i-vector size
3. `ls` :
```
globalAccuPLDA.log : Result for PLDA
globalAccuKNN.log : Result for KNN
globalAccuSVR_Test.log : Result for SVR
globalAccuPerPatientSVR_Test.log : Result for Per Patient SVR
globalAccuEveryoneSVR_Test.log : Result for Everyone SVR
```
  </div>
</details>

The file `runFor.sh` will create the log files with the results of the experiments you ran. The following section explains how to retrieve those results.
 
#### Manually - for one size of i-vector 
The following example will retrieve results for the following i-vector: `trem_noinact_auto30`.

1. `cd <your-path-to-kaldi>/kaldi/egs/beatPDivec/dysk_noinact_auto30/exp/`
2. `cd ivec_650` : Then, choose an i-vector size 
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

1. `cd` to the i-vector location, for example `cd <your-path-to-kaldi>/kaldi/egs/beatPDivec/dysk_noinact_auto30/` was the i-vector used for the [4th submission](https://github.com/Mymoza/BeatPD-CLSP-JHU/wiki/0-Write-Up).
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
    - `sDirFeatsTest` to point to the folder where you have extracted testing features with the AE, `<your-path-to-AE-features>/cis_testing_AE_30ft_orig_inactivity_removed` 
    - `sDirFeatsTrai` to point to the folder where  there is the training data `<your-path-to-AE-features>/AE_30ft_orig_inactivity_removed`
    - `ivecDim` : The i-vector size you are interested in, for the final submission, use `ivecDim=650`. 
<!---
No need to tell this step, it is already provided by default     
- For the number of components, you need to write the components that have been selected as the best for at least one per patient tuning. You will get this info there `cat <your-path-to-kaldi>/kaldi/egs/beatPDivec/dysk_orig_auto60_400fl/exp/ivec_650/globalAccuPerPatientSVR_Test.log` (it is the dictionary you can see at step 7).
-->
4. Go to `CreateCSV_test.ipynb`. We will use the function `generateCSVtest_per_patient` to create a CSV containing test predictions for all subject_ids. 

7. Provide the variables `best_config`, `dest_dir`, and `src_dir`. To reproduce the final submission, simply keep the `best_config` as it is, and replace the paths with yours. The following code show you exactly what you should use:

*Note:* We made the mistake of using the best hyperparameters on `dysk_orig_auto60_400fl` applied to `dysk_noinact_auto30`.
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

dest_dir='<your-path-to-kaldi>/kaldi/egs/beatPDivec/dysk_noinact_auto30/exp/ivec_650/resiVecPerPatientSVR_Fold_all/'
src_dir='<your-path-to-kaldi>/kaldi/egs/beatPDivec/dysk_noinact_auto30/exp/ivec_650/resiVecPerPatientSVR_Fold_all/'

generateCSVtest_per_patient(src_dir, dest_dir, best_config)
```

If you want to experiment with other `best_config` values, the dictionary for best_config is obtained in this file: 
`cat <your-path-to-kaldi>/kaldi/egs/beatPDivec/dysk_noinact_auto30/exp/ivec_650/globalAccuPerPatientSVR_Test.log`

8. Run that cell, and it will create a `csv` file in the provided location `dest_dir`. 

**Note: Submission 4**

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

#### Fusion for test subset of the challenge 

1. Go to the cell under the heading "Example to get predictions file on test subset" 
2. Change the paths
3. Run the cell. There will be an output telling you where your predictions file for dyskinesia was created, like so: 

```
Submission file was created: /BeatPD_predictions/submissionCisPDdyskinesia.csv
```

#### Fusion for Test Folds 
Go to `Dyskinesia - Submission 4 - Average` for an example of how to do fusion evaluation on the test folds. Just give the path to the csv files containing the predictions in `sFilePred1` and `sFilePred2` (obtained [here](#get-preds-trainingtestfolds-perpatient-svr)), like so:
```
sFilePred1='<your-path-to-github-repo>/BeatPD-CLSP-JHU/tsfresh/submit/submission4/kfold_prediction_dyskinesia.csv'
sFilePred2='<your-path-to-kaldi>/kaldi/egs/beatPDivec/dysk_noinact_auto30/exp/ivec_650/resiVecPerPatientSVR_Fold_all/preds_per_patient.csv'
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


<br>
<hr>
<br>


# References 

- [The Biomarker and Endpoint Assessment to Track Parkinson's Disease (BEAT-PD) Challenge](https://www.synapse.org/#!Synapse:syn20825169/wiki/596118)
- Christ, M., Braun, N., Neuffer, J. and Kempa-Liehr A.W. (2018). Time Series FeatuRe Extraction on basis of Scalable Hypothesis tests (tsfresh -- A Python package). Neurocomputing 307 (2018) 72-77, doi:10.1016/j.neucom.2018.03.067. [GitHub](https://github.com/blue-yonder/tsfresh)
- Dehak, Najim, et al. "Front-end factor analysis for speaker verification." IEEE Transactions on Audio, Speech, and Language Processing 19.4 (2010): 788-798. 
