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

## Directory structure 

From `/home/sjoshi/codes/python/BeatPD/data/BeatPD`: 

```
|--cis-pd.ancillary_data : 352 extra files given by the challenge. 
|--cis-pd.ancillary_data.high_pass_mask : Mask with [0,1] of where the high pass filter identified inactivity on ancillary data
|
|--cis-pd.clinical_data : Demographics data about the subjects_id and measurement_id 
|   |
|   |------ CIS-PD_Demographics.csv
|   |------ CIS-PD_UPDRS_Part3.csv
|   |------ CIS-PD_UPDRS_Part1_2_4.csv
|  
|--cis-pd.data_labels
|   |
|   |------ CIS-PD_Ancillary_Data_IDs_Labels.csv
|   |------ CIS-PD_Training_Data_IDs_Labels.csv
|
|
|--cis-pd.training_data : 1858 files - Original training data without any edits
|--cis-pd.training_data.wav_X : Wav files of the training data — the inactivity is NOT removed
|--cis-pd.training_data.wav_Y
|--cis-pd.training_data.wav_Z
|
|--cis-pd.training_data.derivative_original_data : first derivative of accelerometer. High pass filter was also applied so inactivty is removed in these files.  
|
|--cis-pd.training_data.high_pass : Original data where high pass filtered was applied.
|--cis-pd.training_data.high_pass.wav_X : High Pass filtered data to wav files (inactivity is not removed) 
|--cis-pd.training_data.high_pass.wav_Y 
|--cis-pd.training_data.high_pass.wav_Z
|
|--cis-pd.training_data.high_pass_mask : Mask with [0,1] of where the high pass filter identified inactivity 
|--cis-pd.training_data.high_pass_mask.wav_X : Original data where inactivity is removed to wav files 
|--cis-pd.training_data.high_pass_mask.wav_Y 
|--cis-pd.training_data.high_pass_mask.wav_Z
|
|--cis-pd.training_data.k_fold_v1 : Labels divided in 5 folds from which we can read the measurement_id 
|--cis-pd.training_data.k_fold_v2 : Balanced (as much as possible) folds. NaN are replaced with -1 values
|--cis-pd.training_data.k_fold_v3 : Balanced (as much as possible) folds. NaNs are used. 
|
|--cis-pd.training_data.no_silence : Silence removed with pct_change technique 
|
|--cis-pd.testing_data
|
|--real-pd.ancillary_data : Extra data given by the challenge
|--real-pd.ancillary_data.high_pass_mask
|
|--real-pd.clinical_data : Demographics data about the subjects_id and measurement_id 
|--real-pd.data_labels
|   |
|   |------ REAL-PD_Ancillary_Data_IDs_Labels.csv
|   |------ REAL-PD_Training_Data_IDs_Labels.csv
|
|--real-pd.training_data : Original training data without any edits 
|   |
|   |------ smartphone_accelerometer : 526 files
|   |------ smartwatch_accelerometer : 535 files
|   |------ smartwatch_gyroscope : 535 files
|
|--real-pd.training_data.k_fold : Labels divided in 5 folds from which we can read the measurement_id 
|
|-- ubm.dat
|-- gmm.hdf5
```

# Databases

<table class="tg">
  <tr>
    <th class="tg-0pky"></th>
    <th class="tg-0pky">CIS-PD</th>
    <th class="tg-0pky">REAL-PD</th>
  </tr>
  <tr>
    <td class="tg-0pky"># of subject_id training</td>
    <td class="tg-c3ow">16</td>
    <td class="tg-c3ow">12</td>
  </tr>
  <tr>
    <td class="tg-0pky"># of female training</td>
    <td class="tg-c3ow">5</td>
    <td class="tg-c3ow">7</td>
  </tr>
  <tr>
    <td class="tg-0pky"># of male training</td>
    <td class="tg-c3ow">11</td>
    <td class="tg-c3ow">5</td>
  </tr>
  <tr>
    <td class="tg-0pky">Age average (std deviation)</td>
    <td class="tg-c3ow">62.8125 (10.857)</td>
    <td class="tg-c3ow">59.833 (5.828)</td>
  </tr>
</table>


# Where are the features? 
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

Dyskenisia: 
- `dysk_hpf_auto30`
- `dysk_noinact_auto30`
- `dysk_combhpfnoinact_auto30`


## Autoencoder output features 

1. `cd /export/b19/mpgill/BeatPD/`
2. `source activate keras_tf2`

### CIS-PD, training features 

Use `python train_AE.py`
If using raw data, then you don't need to give `my_data_path`.

`$ NO COMMAND YET`

`/export/b03/sbhati/PD/BeatPD/AE_feats` : 30 features, 400 frame length, inactivity not removed. Original data

`$ python train_AE.py --saveAEFeats -dlP '{"remove_inactivity": "True", "my_data_path": "/home/sjoshi/codes/python/BeatPD/data/BeatPD/cis-pd.training_data/", "my_mask_path": "/home/sjoshi/codes/python/BeatPD/data/BeatPD/cis-pd.training_data.high_pass_mask/"}'`

- `AE_30ft_orig_inactivity_removed/` : 30 features, 400 frame length, inactivity removed from original training data

`$ python train_AE.py --saveAEFeats -dlP '{"remove_inactivity": "True", "my_data_path": "/home/sjoshi/codes/python/BeatPD/data/BeatPD/cis-pd.training_data.high_pass/", "my_mask_path": "/home/sjoshi/codes/python/BeatPD/data/BeatPD/cis-pd.training_data.high_pass_mask/"}'`

- `AE_30ft_high_pass_inactivity_removed/` : 30 features, inactivity removed from the high pass filtered training data

`$ python train_AE.py --saveAEFeats -dlP '{"remove_inactivity": "False", "my_data_path": "/home/sjoshi/codes/python/BeatPD/data/BeatPD/cis-pd.training_data.high_pass/", "my_mask_path": "None"}'`

- `AE_30ft_high_pass/` : 30 features, high pass on training data (inactivity is not removed) 

`$ python train_AE.py --latent_dim 60 -dlP '{"remove_inactivity":"False","frame_length":480}' --saveAEFeats --saveFeatDir "/export/b19/mpgill/BeatPD/AE_60ft_480fl_orig/"`

- `AE_60ft_480fl_orig` : 60 ft, frame length 480, original data. Inactivity is not removed.

`$ python train_AE.py --latent_dim 60 -dlP '{"my_mask_path": "/home/sjoshi/codes/python/BeatPD/data/BeatPD/cis-pd.training_data.high_pass_mask/", "remove_inactivity":"True","frame_length":480}' --saveAEFeats --saveFeatDir "/export/b19/mpgill/BeatPD/AE_60ft_480fl_orig_inactivity_removed/"`

- `AE_60ft_480fl_orig_inactivity_removed` : 60 ft, frame length 480, original data. Inactivity is removed. 

`$ python train_AE.py --latent_dim 60 -dlP '{"remove_inactivity":"False","frame_length":320}' --saveAEFeats --saveFeatDir "/export/b19/mpgill/BeatPD/AE_60ft_320fl_orig/"`

- `AE_60ft_320fl_orig` : 60 ft, frame length 320, original data. Inactivity is not removed. 

`$ python train_AE.py --latent_dim 60 -dlP '{"my_mask_path": "/home/sjoshi/codes/python/BeatPD/data/BeatPD/cis-pd.training_data.high_pass_mask/", "remove_inactivity":"True","frame_length":320}' --saveAEFeats --saveFeatDir "/export/b19/mpgill/BeatPD/AE_60ft_320fl_orig_inactivity_removed/"`

- `AE_60ft_320fl_orig_inactivity_removed` : 60 ft, frame length 320, original data. Inactivity is removed. 


### CIS-PD, testing features 

`$ python test_AE.py -dlP '{"my_data_path": "/home/sjoshi/codes/python/BeatPD/data/BeatPD/cis-pd.testing_data/", "my_mask_path": "/home/sjoshi/codes/python/BeatPD/data/BeatPD/cis-pd.testing_data.high_pass_mask/", "remove_inactivity": "True"}' --saveAEFeats`

- `cis_testing_AE_30ft_orig_inactivity_removed`

We won't do the following as they're not obtaining better results : 

`$ python test_AE.py -dlP '{"my_data_path": "/home/sjoshi/codes/python/BeatPD/data/BeatPD/cis-pd.testing_data.high_pass/", "my_mask_path": "/home/sjoshi/codes/python/BeatPD/data/BeatPD/cis-pd.testing_data.high_pass_mask/", "remove_inactivity": "True"}'`

- `cis_testing_AE_30ft_high_pass_inactivity_removed`

`$ python test_AE.py -dlP '{"my_data_path": "/home/sjoshi/codes/python/BeatPD/data/BeatPD/cis-pd.testing_data.high_pass/", "my_mask_path": "/home/sjoshi/codes/python/BeatPD/data/BeatPD/cis-pd.testing_data.high_pass_mask/", "remove_inactivity": "False"}'`

- `cis_testing_AE_30ft_high_pass`

### REAL-PD, training features 

`python train_AE_real.py -dlP '{"sw_acc_data_path":"/home/sjoshi/codes/python/BeatPD/data/BeatPD/real-pd.training_data/smartwatch_accelerometer/", "sw_acc_mask_path":"/home/sjoshi/codes/python/BeatPD/data/BeatPD/real-pd.training_data.high_pass_mask/smartwatch_accelerometer/","sw_gyro_data_path":"/home/sjoshi/codes/python/BeatPD/data/BeatPD/real-pd.training_data/smartwatch_gyroscope/","sw_gyro_mask_path":"/home/sjoshi/codes/python/BeatPD/data/BeatPD/real-pd.training_data.high_pass_mask/smartwatch_gyroscope/", "remove_inactivity": "True"}'  --saveAEFeats`

- `/export/b19/mpgill/BeatPD/real_train_AE_30ft_orig_inactivity_removed/`

`python train_AE_real.py -dlP '{"sw_acc_data_path":"/home/sjoshi/codes/python/BeatPD/data/BeatPD/real-pd.training_data.high_pass/smartwatch_accelerometer/", "sw_acc_mask_path":"/home/sjoshi/codes/python/BeatPD/data/BeatPD/real-pd.training_data.high_pass_mask/smartwatch_accelerometer/","sw_gyro_data_path":"/home/sjoshi/codes/python/BeatPD/data/BeatPD/real-pd.training_data.high_pass/smartwatch_gyroscope/","sw_gyro_mask_path":"/home/sjoshi/codes/python/BeatPD/data/BeatPD/real-pd.training_data.high_pass_mask/smartwatch_gyroscope/", "remove_inactivity": "True"}'  --saveAEFeats`

- /export/b19/mpgill/BeatPD/real_train_AE_30ft_high_pass_inactivity_removed/

`$ python train_AE_real.py -dlP '{"sw_acc_data_path":"/home/sjoshi/codes/python/BeatPD/data/BeatPD/real-pd.training_data.high_pass/smartwatch_accelerometer/", "sw_acc_mask_path":"/home/sjoshi/codes/python/BeatPD/data/BeatPD/real-pd.training_data.high_pass_mask/smartwatch_accelerometer/","sw_gyro_data_path":"/home/sjoshi/codes/python/BeatPD/data/BeatPD/real-pd.training_data.high_pass/smartwatch_gyroscope/","sw_gyro_mask_path":"/home/sjoshi/codes/python/BeatPD/data/BeatPD/real-pd.training_data.high_pass_mask/smartwatch_gyroscope/",  "remove_inactivity": "False"}'  --saveAEFeats`

- `/export/b19/mpgill/BeatPD/real_train_AE_30ft_high_pass/` : 30 features, high pass on training data (inactivity is not removed)

(`sw_acc_data_path`, `sw_acc_mask_path`, `sw_gyro_data_path` and `sw_gyro_mask_path` are not necessary in the following command as these are their default values) 

- `$ python train_AE_real.py --latent_dim 60 -dlP '{"sw_acc_data_path":"/home/sjoshi/codes/python/BeatPD/data/BeatPD/real-pd.training_data/smartwatch_accelerometer/", "sw_acc_mask_path":"/home/sjoshi/codes/python/BeatPD/data/BeatPD/real-pd.training_data.high_pass_mask/smartwatch_accelerometer/","sw_gyro_data_path":"/home/sjoshi/codes/python/BeatPD/data/BeatPD/real-pd.training_data/smartwatch_gyroscope/","sw_gyro_mask_path":"/home/sjoshi/codes/python/BeatPD/data/BeatPD/real-pd.training_data.high_pass_mask/smartwatch_gyroscope/", "remove_inactivity": "False", "frame_length":480}'  --saveAEFeats --saveFeatDir "/export/b19/mpgill/BeatPD/real_train_AE_60ft_480fl_orig/"`

- `real_train_AE_60ft_480fl_orig`

- `$ python train_AE_real.py --latent_dim 60 -dlP '{"sw_acc_data_path":"/home/sjoshi/codes/python/BeatPD/data/BeatPD/real-pd.training_data/smartwatch_accelerometer/", "sw_acc_mask_path":"/home/sjoshi/codes/python/BeatPD/data/BeatPD/real-pd.training_data.high_pass_mask/smartwatch_accelerometer/","sw_gyro_data_path":"/home/sjoshi/codes/python/BeatPD/data/BeatPD/real-pd.training_data/smartwatch_gyroscope/","sw_gyro_mask_path":"/home/sjoshi/codes/python/BeatPD/data/BeatPD/real-pd.training_data.high_pass_mask/smartwatch_gyroscope/", "remove_inactivity": "True", "frame_length":480}'  --saveAEFeats --saveFeatDir "/export/b19/mpgill/BeatPD/real_train_AE_60ft_480fl_orig_inactivity_removed/"`

- `real_train_AE_60ft_480fl_orig_inactivity_removed`  

- `$ python train_AE_real.py --latent_dim 60 -dlP '{"sw_acc_data_path":"/home/sjoshi/codes/python/BeatPD/data/BeatPD/real-pd.training_data/smartwatch_accelerometer/", "sw_acc_mask_path":"/home/sjoshi/codes/python/BeatPD/data/BeatPD/real-pd.training_data.high_pass_mask/smartwatch_accelerometer/","sw_gyro_data_path":"/home/sjoshi/codes/python/BeatPD/data/BeatPD/real-pd.training_data/smartwatch_gyroscope/","sw_gyro_mask_path":"/home/sjoshi/codes/python/BeatPD/data/BeatPD/real-pd.training_data.high_pass_mask/smartwatch_gyroscope/", "remove_inactivity": "False", "frame_length":320}'  --saveAEFeats --saveFeatDir "/export/b19/mpgill/BeatPD/real_train_AE_60ft_320fl_orig/"`

- `real_train_AE_60ft_320fl_orig`  

- `$ python train_AE_real.py --latent_dim 60 -dlP '{"sw_acc_data_path":"/home/sjoshi/codes/python/BeatPD/data/BeatPD/real-pd.training_data/smartwatch_accelerometer/", "sw_acc_mask_path":"/home/sjoshi/codes/python/BeatPD/data/BeatPD/real-pd.training_data.high_pass_mask/smartwatch_accelerometer/","sw_gyro_data_path":"/home/sjoshi/codes/python/BeatPD/data/BeatPD/real-pd.training_data/smartwatch_gyroscope/","sw_gyro_mask_path":"/home/sjoshi/codes/python/BeatPD/data/BeatPD/real-pd.training_data.high_pass_mask/smartwatch_gyroscope/", "remove_inactivity": "True", "frame_length":480}'  --saveAEFeats --saveFeatDir "/export/b19/mpgill/BeatPD/real_train_AE_60ft_320fl_orig_inactivity_removed/"`

- `real_train_AE_60ft_320fl_orig_inactivity_removed` 


### REAL-PD, testing features 

`$ python test_AE_real.py -dlP '{"sw_acc_data_path":"/home/sjoshi/codes/python/BeatPD/data/BeatPD/real-pd.testing_data/smartwatch_accelerometer/", "sw_acc_mask_path":"/home/sjoshi/codes/python/BeatPD/data/BeatPD/real-pd.testing_data.high_pass_mask/smartwatch_accelerometer/","sw_gyro_data_path":"/home/sjoshi/codes/python/BeatPD/data/BeatPD/real-pd.testing_data/smartwatch_gyroscope/","sw_gyro_mask_path":"/home/sjoshi/codes/python/BeatPD/data/BeatPD/real-pd.testing_data.high_pass_mask/smartwatch_gyroscope/", "remove_inactivity": "True"}'  --saveAEFeats`

- `/export/b19/mpgill/BeatPD/real_testing_AE_30ft_orig_inactivity_removed/`

`$ python test_AE_real.py -dlP '{"sw_acc_data_path":"/home/sjoshi/codes/python/BeatPD/data/BeatPD/real-pd.testing_data.high_pass/smartwatch_accelerometer/", "sw_acc_mask_path":"/home/sjoshi/codes/python/BeatPD/data/BeatPD/real-pd.testing_data.high_pass_mask/smartwatch_accelerometer/","sw_gyro_data_path":"/home/sjoshi/codes/python/BeatPD/data/BeatPD/real-pd.testing_data.high_pass/smartwatch_gyroscope/","sw_gyro_mask_path":"/home/sjoshi/codes/python/BeatPD/data/BeatPD/real-pd.testing_data.high_pass_mask/smartwatch_gyroscope/", "remove_inactivity": "True"}'  --saveAEFeats`

- `real_testing_AE_30ft_high_pass_inactivity_removed`

`$ python test_AE_real.py -dlP '{"sw_acc_data_path":"/home/sjoshi/codes/python/BeatPD/data/BeatPD/real-pd.testing_data.high_pass/smartwatch_accelerometer/", "sw_acc_mask_path":"/home/sjoshi/codes/python/BeatPD/data/BeatPD/real-pd.testing_data.high_pass_mask/smartwatch_accelerometer/","sw_gyro_data_path":"/home/sjoshi/codes/python/BeatPD/data/BeatPD/real-pd.testing_data.high_pass/smartwatch_gyroscope/","sw_gyro_mask_path":"/home/sjoshi/codes/python/BeatPD/data/BeatPD/real-pd.testing_data.high_pass_mask/smartwatch_gyroscope/", "remove_inactivity": "False"}'  --saveAEFeats`

- `real_testing_AE_30ft_high_pass` : inactivity is not removed


# Visualization 

TODO 

# Step-By-Step guide 

### CIS-PD: Create High Pass Data
TODO 

### Create Masks for inactivity removal 
Masks were created in the notebook `analyze_data_cleaned.ipynb`, like so: 
```
remove_inactivity_highpass(
    df_train_label,
    path_train_data,
    data_type,
    energy_threshold=5,
    duration_threshold=3000,
    plot_frequency_response=False,
    mask_path='/home/sjoshi/codes/python/BeatPD/data/BeatPD/cis-pd.'+
    data_subset+'.high_pass_mask/')
```
Two parameters can be tuned:
* `energy_threshold` : what percentage of the max energy do we consider as inactivity? The current masks generated have used the threshold of 5% 
* `duration_threshold` : how long do we want to have inactivity before we remove it? For example 3000x0.02ms=1min of inactivity minimum before those candidates are considered inactivty and will be removed. 

### Evaluation steps 

#### Automatisation to generate the results 

To get all the results for all the combinations of `ivecDim` for every class (`on/off`, `tremor`, `dysk`) for the SVR model, use this script:
1. `./run_SVR_pkl_files.sh`
2. `./run_all_evaluation_SVR.sh`

#### Manually 
1. To create the pkl files that are going to let you get the challenge final score afterward: 

- `./runSVRFold.sh $sOut $ivecDim $sDirFeats`
- `./runKNNFold.sh $sOut $ivecDim $sDirFeats`

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


2. Get the final score as used in the challenge (weighted MSE): 

- `./evaluate_global_SVR.sh /export/c08/lmorove1/kaldi/egs/beatPDivec/v1_autoenc/exp/ivec_50/ /export/c08/lmorove1/kaldi/egs/beatPDivec/v1_autoenc/exp/ivec_50/`

- `./evaluate_global_acc_knn.sh /export/c08/lmorove1/kaldi/egs/beatPDivec/trem_noinact_auto30/exp/ivec_350/ /export/c08/lmorove1/kaldi/egs/beatPDivec/trem_noinact_auto30/exp/ivec_350/`

This script will generate a `.log` file from the name and location provided in `evaluate_global_acc.sh`, like so:

```
$cmd $sOut/globalAccuSVR_Test.log \
     ${filePath}get_final_scores_accuracy.py  --file-path $sFileTrai \
     --is-svr
```
The result will be stored in `/export/c08/lmorove1/kaldi/egs/beatPDivec/v1_autoenc/exp/ivec_50/globalAccuSVR_Test.log`

To get a final score for KNN, only add the `--is-knn` flag, like so: 

```
$cmd $sOut/globalAccuKNN_Test.log \
     ${filePath}get_final_scores_accuracy.py  --file-path $sFileTrai \
     --is-knn
```

# Inactivity (apply high-pass filter) 

## Example of what it does with plots 

Here's an example for measurement_id `db2e053a-0fb8-4206-891a-6f079fb14e3a` from the CIS-PD database.

<img src="images/initial-plot-accelerometer.png" width="500">


After the High pass filter (inactivity identified is filled with X,Y,Z=0 for the purpose of the plot) :

<img src="images/plot-after-highpass.png" width="400">

It looks good, with a straight line of inactivity on “zero”… However, it’s not  visible to the eyes, but there are some values left at the complete beginning of the dataframe from index 0 to index 31. 

Then we have inactivity from index 32 to 26073.

<img src="images/table-explanation-why-not-perfect.png" width="300">

This explains why the accelerometer with inactivity removed looks like this: 

We have 32 values right at the beginning which prevents the graph to show just the [600,1200] part

<img src="images/final-plot-inactivity-removed.png" width="400">

## How to remove inactivity

Masks have already been created detecting inactivity for all the databases. They are stored in the `*.high_pass_mask` folder. 


What's left is to apply the mask. To do so, a function called `apply_mask` located in `transform_data.py` can be used. 

```
# import transform_data
from transform_data import apply_mask

# path_train_data : path to the original training files which we want to apply the highpass filter on 
# measurement_id : measurement_id we want to apply the mask to
# mask_path: Path where to apply the mask to the wav file 

df_train_data = apply_mask(path_train_data,
                                   measurement_id,
                                   mask_path)
```

# Working in Jupyter Notebooks 

If you're working in Jupyter notebooks, you will probably need to import functions from python files. 

You should use these two lines to make sure that if you make changes to the python files, the code that is being called from your Jupyter Notebook will be updated: 

```
%load_ext autoreload
%autoreload 2

from transform_data import *
from create_graphs import *
```

