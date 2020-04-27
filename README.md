# BEATPD 

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



# Where are the i-vectors? 

The directory of the ivectors are all reported in our [Google Spreadsheet](https://docs.google.com/spreadsheets/d/11l7S49szMllpebGg2gji2aBea35iqLqO5qrlOBSJnIc/) presenting our results for the different experiments. 

## MFCC 
`cd /export/c08/lmorove1/kaldi/egs/beatPDivec/*/exp/ivectors_Training_Fold0/ivector.scp`
- `/v1/*/*/ivector.scp`:  on/off using the x axis and 20 mfcc

- `v1_3ax/exp3x/` : on/off using the three axis and 10 mfcc 
- `v1_3ax_10mfcc_dysk/exp3x/` : dysk using the three axis and 10 mfcc
- `v1_3ax_10mfcc_tr/exp3x/`: tremor using the three axis and 10 mfcc

- `v1_autoenc` : on/off using the three axis and autoencoder (30 ft AE) 
- `v1_dysk_auto` : dyskenisia using the three axis and autoencoder (30ft AE)
- `v1_trem_auto` : tremor using the three axis and autoencoder (30ft AE)

## Autoencoder 

On/Off: 
- `on_off_hpf_auto30` : High Pass filtered data. Inactivity is not removed.
- `on_off_noinact_auto30`: Inactivity removed on original training data. 30 fts.
- `on_off_combhpfnoinact_auto30`: High Pass filtered data. Inactivity is removed. 

Tremor: 
- `trem_hpf_auto30` 
- `trem_noinact_auto30`
- `trem_combhpfnoinact_auto30`

Dyskinesia: 
- `dysk_hpf_auto30`
- `dysk_noinact_auto30`
- `dysk_combhpfnoinact_auto30`


# Step-By-Step guide 

This step-by-step guide will cover the following steps: 

1. [Prepare the data](#prepare-data)
2. [Embeddings](#embeddings)
    1. [MFCC](#2.1-mfcc) 
    2. AutoEncoder (AE)
        1. Train the AutoEncoder
        2. Save AE Features
    3. Create i-vectors
    4. Get results for SVR/SVR Per Patient/SVR Everyone
3. TSFRESH + XGBOOST 
4. Fusion 

<a name="prepare-data"></a>
## 1. Prepare the data 

All the steps to prepare the data is done in the Jupyter Notebook `prepare_data.ipynb`. 

1. Open the notebook
2. Change the `data_dir` variable for the absolute path to the folder that contains the data given by the challenge. In this folder, you should already have the following directories: 
```
/export/b19/mpgill/BeatPD_data  $ ls
cis-pd.ancillary_data  cis-pd.testing_data   real-pd.ancillary_data  real-pd.testing_data
cis-pd.data_labels     cis-pd.training_data  real-pd.data_labels     real-pd.training_data
```
3. Execute the cells in the Notebook. It will create several folders needed to reproduce the experiments. 

<a name="embeddings"></a>
## 2. Embeddings 

<a name="2.1-mfcc"></a>
### 2.1 MFCC 

🙅‍♀️: This section hasn't been written yet. It is not a priority as MFCCs did not provide best results and they were not used for submission. 

### 2.2 AutoEncoder features 

#### 2.2.1 Train the AutoEncoder 

🛑TODO: - Ask Bhati for help. How to train the AutoEncoder? 
Code needs to be in github 

🛑TODO: How to Create the keras_tf environment

#### 2.2.2 Get AutoEncoder (AE) Features 

1. `git checkout ml_dl`: The code to get features from the AutoEncoder is in another branch. 
2. `cd ml_dl`
3. `source activate keras_tf2`
4. Go to this [wiki page](https://github.com/Mymoza/BeatPD-CLSP-JHU/wiki/3--Creating-AutoEncoder-Features#create-autoencoder-features) that lists many examples of commands you can use the create the required AE features.

### 2.2.3 Create i-vectors 

🛑TODO: "Create i-vectors": Correct vocabulary? 

After creating Autoencoder features, we can create i-vectors.

You need to have [Kaldi](https://kaldi-asr.org/doc/install.html) installed first. Follow Kaldi's instructions to install. 

🛑TODO: Ask Laureano how he suggest to do this from GitHub 
🛑TODO: "This is where all the ivectors will be created" : is it the correct vocabulary? 

The following steps will vary a lot depending on what ivector you want to create. One way to decide which ivector to create is to view the [Google spreadsheet results](https://docs.google.com/spreadsheets/d/11l7S49szMllpebGg2gji2aBea35iqLqO5qrlOBSJnIc/edit?usp=sharing) and find out which result you are interested in replicating. The column "C" has notes for each appropriate cell with the name of the ivector folder we use. You can use the same nomenclature to replicate our experiments. 

1. `cd /export/c08/lmorove1/kaldi/egs/beatPDivec` : This is where all the ivectors will be created 
2. `mkdir *****` : Create a folder with a meaningful name about the ivectors we want to create
3. `cd ****`
4. `mkdir data`
5. `cp -rf /export/c08/lmorove1/kaldi/egs/beatPDivec/default_data/v2_auto/. ./`
6. `cp -rf /export/c08/lmorove1/kaldi/egs/beatPDivec/default_data/autoencData/****/. data/.` 
Replace "****" with either `on_off`, `trem` or `dysk`
7. `cp ../on_off_noinact_auto60_480fl/run_auto.sh .` I use `on_off_noinact_auto60_480fl/run_auto.sh` only because we made a few changes to the one copied over from step 5 to make it faster on the grid. We also removed KNN and PLDA step as at this time we're focusing on SVR results. 
8. `cp ../on_off_noinact_auto60_400fl/runFor.sh .` Copy a `runFor` from another folder as it's not being copied over from step 5. Be careful that the folder you decide to copy it from as the `local/evaluate SVR` line inside. 
9. In `runFor.sh`, change the `sDirFeats` variable pointing to a folder of AutoEncoder features
10. `screen -R name_of_your_screen`
11. `cd /export/c08/lmorove1/kaldi/egs/beatPDivec/****`
12. `qsub -l mem_free=30G,ram_free=30G -pe smp 6 -cwd -e /export/b19/mpgill/errors/errors_dysk_orig_auto60_400fl -o /export/b19/mpgill/outputs/outputs_dysk_orig_auto60_400fl runFor.sh`

🛑TODO: Good vocabulary? 
Launching the `runFor.sh` file will launch the i-vectors / UBM extraction, as well as the KNN/SVRs schemes. 

### 2.2.4 Get results 

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

### Evaluation steps 

#### Automatisation to generate the results 

🛑TODO: To update

To get all the results for all the combinations of `ivecDim` for every class (`on/off`, `tremor`, `dysk`) for the SVR model, use this script:
1. `./run_SVR_pkl_files.sh` 
2. `./run_all_evaluation_SVR.sh`

#### Manually 
1. To create the pkl files that are going to let you get the challenge final score afterward: 

- `./runSVRFold.sh $sOut $ivecDim $sDirFeats`
- `./runKNNFold.sh $sOut $ivecDim $sDirFeats`
- `./runEveryoneSVRFold.sh ${sOut} $ivecDim $sDirFeats`

Or simply use a script like this to automate the ivectors dimension for a provided folder of features: 

```
echo Working on tremor

#sOut=/export/c08/lmorove1/kaldi/egs/beatPDivec/v1_autoenc/exp/
#sDirFeats=/export/c08/lmorove1/kaldi/egs/beatPDivec/v1_autoenc

sOut=/export/c08/lmorove1/kaldi/egs/beatPDivec/trem_noinact_auto30/exp/
sDirFeats=/export/c08/lmorove1/kaldi/egs/beatPDivec/trem_noinact_auto30

for ivecDim in 50 100 150 200 250 300 350 400 450 500 550; do
    echo Working on ${ivecDim}
    ./runKNNFold.sh ${sOut} $ivecDim $sDirFeats
done
```
Many examples are available on the repository in the folder `run_SVR/`. 

2. If pickle files are already created and you just want to get the final score as used in the challenge (weighted MSE): 

#### For KNN 

- `./evaluate_global_acc_knn.sh /export/c08/lmorove1/kaldi/egs/beatPDivec/trem_noinact_auto30/exp/ivec_350/ /export/c08/lmorove1/kaldi/egs/beatPDivec/trem_noinact_auto30/exp/ivec_350/`

To get a final score for KNN, only add the `--is-knn` flag, like so: 

```
$cmd $sOut/globalAccuKNN_Test.log \
     ${filePath}get_final_scores_accuracy.py  --file-path $sFileTrai \
     --is-knn
```

#### For SVR 

- `./evaluate_global_SVR.sh /export/c08/lmorove1/kaldi/egs/beatPDivec/v1_autoenc/exp/ivec_50/ /export/c08/lmorove1/kaldi/egs/beatPDivec/v1_autoenc/exp/ivec_50/`

_FIXME: Is it really that sh file?_ 

This script will generate a `.log` file from the name and location provided in `evaluate_global_acc.sh`, like so:

```
$cmd $sOut/globalAccuSVR_Test.log \
     ${filePath}get_final_scores_accuracy.py  --file-path $sFileTrai \
     --is-svr
```

The result will be stored in `/export/c08/lmorove1/kaldi/egs/beatPDivec/v1_autoenc/exp/ivec_50/globalAccuSVR_Test.log`

#### For SVR Per Patient 

The SVR Per Patient uses the pkl files created for the original SVR. That means it uses pkl files that were created per configuration, but for all patients. 

These files provide the MSE per patient for each configuration.

To get results, it's actually in the file `get_final_scores_accuracy.py`. These is the function called `get_final_scores_SVR_lowest_mse_for_subjectid` that gets final Scores for the individual SVR but with using different parameters for each patient depending on what set of parameters worked best for that patient.

##### Option 1 (the best one)

Use the `runFor.sh` file in the repository. For example:

`cd /export/c08/lmorove1/kaldi/egs/beatPDivec/dysk_noinact_auto30/`

If you don't need to run the extractions of ivectors and other stuff, make sure to comment some code in the `runFor.sh` file before running it. 

`qsub -l mem_free=30G,ram_free=30G -pe smp 6 -cwd -e /export/b19/mpgill/errors/errors_dysk_noinact_auto30 -o /export/b19/mpgill/outputs/outputs_dysk_noinact_auto30 runFor.sh`

`vim runFor.sh` : Edit this file to run stage 5. You can comment the part that extracts ivectors if that's already done. 

Make sure `local/evaluate_global_per_patient_SVR.sh` is ran also following the `run_auto.sh` stage 5 to create the log files with results (all done in the `runFor.sh` file). 

##### Option 2 (in your own repo, on a specific branch for example)

For this type of results, pkl files are the same as the one for regular SVR. When `.pkl` files are present in `resiVecSVR_Fold*` folders, than you can find the best set of hyperparameters with a function in the file `get_final_scores_accuracy.py` and get a final score. 

You need to edit the file `run_evaluate_global_per_patient_SVR.sh` and write manually the name of the folder with the ivectors you want to get results for, as well as for what dimensions of ivectors as it will loop over all the ivector dimensions given. 

Here's a command as an example: 

```
qsub -l mem_free=30G,ram_free=30G -pe smp 6 -cwd -e /export/b19/mpgill/errors/error_file -o /export/b19/mpgill/outputs/output_file run_evaluate_global_per_patient_SVR.sh
```

The file `run_evaluate_global_per_patient_SVR.sh` actually calls the file `evaluate_global_per_patient_SVR.sh` that creates the log files with the results we are looking for : `/export/c08/lmorove1/kaldi/egs/beatPDivec/***/exp/ivec_***/globalAccuPerPatientSVR_Test.log`

##### Option 3 (very manual)

If you only want to have results for one ivector dimension, you can do this directly like this: 

`./evaluate_global_per_patient_SVR_patient_SVR.sh /export/c08/lmorove1/kaldi/egs/beatPDivec/v1_autoenc/exp/ivec_50/ /export/c08/lmorove1/kaldi/egs/beatPDivec/v1_autoenc/exp/ivec_50/` 

#### For SVR Everyone 

##### Option 1 (the best one)

To create `.pkl` files for SVR Everyone, `stage=6` needs to be ran. 

`cd /export/c08/lmorove1/kaldi/egs/beatPDivec/v1_autoenc/exp/ivec_50/`

`vim runFor.sh` : Edit this file to run stage 6. You can comment the part that extracts ivectors if that's already done. 

Make sure `local/evaluate_global_everyone_SVR.sh` is ran also following the `run_auto.sh` stage 6 to create the log files with results. 

##### Option 3 

To get results: 
`./evaluate_global_everyone_SVR.sh /export/c08/lmorove1/kaldi/egs/beatPDivec/v1_autoenc/exp/ivec_50/ /export/c08/lmorove1/kaldi/egs/beatPDivec/v1_autoenc/exp/ivec_50/`



# Get Predictions 

### Per Patient SVR 

#### Option 1 (manual - a bit bad way)

1. Open the notebook `drafts.ipynb` 
2. Go to the section [Get Predictions Per Patient SVR](http://localhost:6099/notebooks/drafts_and_tests.ipynb#Get-Predictions-for-Per-Patient-SVR)
3. For the subchallenge of your choice, like `Dysk Best Config`, change the variables to point to the best config you want to get predictions for. Change the folder of the features (in this case, it is `dysk_orig_auto60_400fl`) and the ivectors dimension (in this case it is `650`). 

```
sFileTrai="/export/c08/lmorove1/kaldi/egs/beatPDivec/dysk_orig_auto60_400fl/exp/ivec_650/ivectors_Training_Fold"+fold+"/ivector.scp"
    sFileTest="/export/c08/lmorove1/kaldi/egs/beatPDivec/dysk_orig_auto60_400fl/exp/ivec_650/ivectors_Testing_Fold"+fold+"/ivector.scp"

sOut="/export/c08/lmorove1/kaldi/egs/beatPDivec/dysk_orig_auto60_400fl/exp/ivec_650/resiVecPerPatientSVR_Fold"+fold
```

4. Provide the `components` and `c value` you want to create pkl files for. In an ideal world, we would be able to provide for each patient their configuration we're interested in getting. At this time unfortunately, each configuration provided will create pkl files for all patients and not just the one we are interested in. You will know what configurations you need to provide by looking at the log file for the configuration, like this : 

```
------ GLOBAL WINNER PARAMETERS ------
{1004: ['/objs_450_kernel_linear_c_0.002_eps_0.1.pkl', 1.1469489658686098],
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
Train Final score :  0.06395586325048086
Test Final score :  0.4771436603152803
```

5. Run that cell. It will create individual pkl files for each patient containing the predictions provided. 

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

8. Run that cell, and it will create a `csv` file in the provided location `dst_dir`. 

# Working in Jupyter Notebooks 

## Import functions 

If you're working in Jupyter notebooks, you will probably need to import functions from python files. 

You should use these two lines to make sure that if you make changes to the python files, the code that is being called from your Jupyter Notebook will be updated: 

```
%load_ext autoreload
%autoreload 2

from transform_data import *
from create_graphs import *
```

## Opening Jupyter notebooks on the grid - example 

1. `ssh -L 8805:b19:8805 -J mpgill@login.clsp.jhu.edu mpgill@b19`
2. `screen -R marie-jup`
3. `cd /home/sjoshi/codes/python/BeatPD`
4. `jupyter-notebook --no-browser --port 8805`
