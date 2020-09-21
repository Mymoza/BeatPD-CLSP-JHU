#!/bin/bash

###########################################################
# This file is used to get predictions on the test folds
###########################################################

conda activate BeatPD_xgboost

path_labels_cis=/home/sjoshi/codes/python/BeatPD/data/BeatPD/cis-pd.data_labels/
path_labels_real=/home/sjoshi/codes/python/BeatPD/data/BeatPD/real-pd.data_labels/


# mkdir xgb_combine_5_rotation
mkdir xgb_comb_rotate_noise
# msek_path="$(pwd)/xgb_combine_5_rotation/"
msek_path="$(pwd)/xgb_comb_rotate_noise/"


# CIS-PD
# python src/gridsearch.py on_off features/cis-pd.training.csv ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv 0
# python src/gridsearch.py tremor features/cis-pd.training.csv ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv 0 
# python src/gridsearch.py dyskinesia features/cis-pd.training.csv ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv 0

# python test_before.py on_off --features features/cis-pd.training.csv --features features/cis-pd.training.combhpfnoinact.noise_mu_0_sig_0.1.csv --labels ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv 

# echo ----- Start cis-pd.training ------

# python src/gridsearch.py on_off --features features/cis-pd.training.csv \
#                                 --labels ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv \
#                                 --filename orig \
#                                 --pred_path ${msek_path} \
#                                 --msek_path ${msek_path}

# python src/gridsearch.py tremor --features features/cis-pd.training.csv \
#                                 --labels ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv \
#                                 --filename orig \
#                                 --pred_path ${msek_path} \
#                                 --msek_path ${msek_path}

# python src/gridsearch.py dyskinesia --features features/cis-pd.training.csv \
#                                     --labels ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv \
#                                     --filename orig \
#                                     --pred_path ${msek_path} \
#                                     --msek_path ${msek_path}

# echo ----- End cis-pd.training ------

# echo ----- Start cis-pd.training.combhpfnoinact ------

# python src/gridsearch.py on_off --features features/cis-pd.training.combhpfnoinact.csv \
#                                 --labels ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv \
#                                 --filename combhpfnoinact \
#                                 --pred_path ${msek_path} \
#                                 --msek_path ${msek_path}

# python src/gridsearch.py tremor --features features/cis-pd.training.combhpfnoinact.csv \
#                                 --labels ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv \
#                                 --filename combhpfnoinact \
#                                 --pred_path ${msek_path} \
#                                 --msek_path ${msek_path}


# python src/gridsearch.py dyskinesia --features features/cis-pd.training.combhpfnoinact.csv \
#                                     --labels ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv \
#                                     --filename combhpfnoinact \
#                                     --pred_path ${msek_path} \
#                                     --msek_path ${msek_path}

# echo ----- End cis-pd.training.combhpfnoinact ------

# echo ----- Resample Data Augmentation ----- 

# echo -- Start 0.9 -- 

# echo 1. original on/off
# python src/gridsearch.py on_off --features features/cis-pd.training.csv \
#                                 --features features/cis-pd.training.resample_0.9.csv \
#                                 --labels ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv \
#                                 --filename original_resample_0.9

# echo 2. original tremor
# python src/gridsearch.py tremor --features features/cis-pd.training.csv \
#                                 --features features/cis-pd.training.resample_0.9.csv \
#                                 --labels ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv \
#                                 --filename original_resample_0.9

# echo 3. original dysk
# python src/gridsearch.py dyskinesia --features features/cis-pd.training.csv \
#                                     --features features/cis-pd.training.resample_0.9.csv \
#                                     --labels ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv \
#                                     --filename original_resample_0.9

# echo -- End 0.9 -- 

# echo -- Start 1.1 --

# echo 1. original on/off
# python src/gridsearch.py on_off --features features/cis-pd.training.csv \
#                                 --features features/cis-pd.training.resample_1.1.csv \
#                                 --labels ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv \
#                                 --filename original_resample_1.1

# echo 2. original tremor
# python src/gridsearch.py tremor --features features/cis-pd.training.csv \
#                                 --features features/cis-pd.training.resample_1.1.csv \
#                                 --labels ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv \
#                                 --filename original_resample_1.1

# echo 3. original dysk
# python src/gridsearch.py dyskinesia --features features/cis-pd.training.csv \
#                                     --features features/cis-pd.training.resample_1.1.csv \
#                                     --labels ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv \
#                                     --filename original_resample_1.1

# echo -- End 1.1 --


# echo -- Start 0.9 -- 

# echo 4. combhpfnoinact on_off
# python src/gridsearch.py on_off --features features/cis-pd.training.combhpfnoinact.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.resample_0.9.csv \
#                                 --labels ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv \
#                                 --filename combhpfnoinact_resample_0.9
# echo 5. combhpfnoinact tremor
# python src/gridsearch.py tremor --features features/cis-pd.training.combhpfnoinact.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.resample_0.9.csv \
#                                 --labels ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv \
#                                 --filename combhpfnoinact_resample_0.9 \
#                                 --pred_path ${msek_path} \
#                                 --msek_path ${msek_path}
# echo 6. combhpfnoinact dysk
# python src/gridsearch.py dyskinesia --features features/cis-pd.training.combhpfnoinact.csv \
#                                     --features features/cis-pd.training.combhpfnoinact.resample_0.9.csv \
#                                     --labels ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv \
#                                     --filename combhpfnoinact_resample_0.9 \
#                                     --pred_path ${msek_path} \
#                                     --msek_path ${msek_path}

# echo -- End 0.9 -- 

# echo -- Start 1.1 -- 

# echo 4. combhpfnoinact on_off
# python src/gridsearch.py on_off --features features/cis-pd.training.combhpfnoinact.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.resample_1.1.csv \
#                                 --labels ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv \
#                                 --filename combhpfnoinact_resample_1.1
# echo 5. combhpfnoinact tremor
# python src/gridsearch.py tremor --features features/cis-pd.training.combhpfnoinact.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.resample_1.1.csv \
#                                 --labels ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv \
#                                 --filename combhpfnoinact_resample_1.1 \
#                                 --pred_path ${msek_path} \
#                                 --msek_path ${msek_path}
# echo 6. combhpfnoinact dysk
# python src/gridsearch.py dyskinesia --features features/cis-pd.training.combhpfnoinact.csv \
#                                     --features features/cis-pd.training.combhpfnoinact.resample_1.1.csv \
#                                     --labels ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv \
#                                     --filename combhpfnoinact_resample_1.1 \
#                                     --pred_path ${msek_path} \
#                                     --msek_path ${msek_path}

# echo -- End 1.1 -- 

# # echo ----- End Resample Data Augmentation ----- 

# # exit 0 


# echo ------ Noise Data Augmentation ------ 
# echo 1. original on/off

# python src/gridsearch.py on_off --features features/cis-pd.training.csv \
#                                 --features features/cis-pd.training.noise_mu_0_sig_0.1.csv \
#                                 --labels ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv \
#                                 --filename original.noise_mu_0_sig_0.1

# echo 2. original tremor
# python src/gridsearch.py tremor --features features/cis-pd.training.csv \
#                                 --features features/cis-pd.training.noise_mu_0_sig_0.1.csv \
#                                 --labels ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv \
#                                 --filename original.noise_mu_0_sig_0.1

# echo 3. original dysk
# python src/gridsearch.py dyskinesia --features features/cis-pd.training.csv \
#                                     --features features/cis-pd.training.noise_mu_0_sig_0.1.csv \
#                                     --labels ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv \
#                                     --filename original.noise_mu_0_sig_0.1

# echo 4. combhpfnoinact on_off
# python src/gridsearch.py on_off --features features/cis-pd.training.combhpfnoinact.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.noise_mu_0_sig_0.1.csv \
#                                 --labels ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv \
#                                 --filename combhpfnoinact.noise_mu_0_sig_0.1
# echo 5. combhpfnoinact tremor
# python src/gridsearch.py tremor --features features/cis-pd.training.combhpfnoinact.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.noise_mu_0_sig_0.1.csv \
#                                 --labels ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv \
#                                 --filename combhpfnoinact.noise_mu_0_sig_0.1 \
#                                 --pred_path ${msek_path} \
#                                 --msek_path ${msek_path}
# echo 6. combhpfnoinact dysk
# python src/gridsearch.py dyskinesia --features features/cis-pd.training.combhpfnoinact.csv \
#                                     --features features/cis-pd.training.combhpfnoinact.noise_mu_0_sig_0.1.csv \
#                                     --labels ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv \
#                                     --filename combhpfnoinact.noise_mu_0_sig_0.1 \
#                                     --pred_path ${msek_path} \
#                                     --msek_path ${msek_path}
# echo ------ End Noise Data Augmentation ------ 

# echo -- Start Rotation 1 --

# echo 1. original on/off
# python src/gridsearch.py on_off --features features/cis-pd.training.csv \
#                                 --features features/cis-pd.training.rotate_1.csv \
#                                 --labels ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv \
#                                 --filename original_rotate_1 \
#                                 --pred_path ${msek_path} \
#                                 --msek_path ${msek_path}

# echo 2. original tremor
# python src/gridsearch.py tremor --features features/cis-pd.training.csv \
#                                 --features features/cis-pd.training.rotate_1.csv \
#                                 --labels ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv \
#                                 --filename original_rotate_1 \
#                                 --pred_path ${msek_path} \
#                                 --msek_path ${msek_path}

# echo 3. original dysk
# python src/gridsearch.py dyskinesia --features features/cis-pd.training.csv \
#                                     --features features/cis-pd.training.rotate_1.csv \
#                                     --labels ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv \
#                                     --filename original_rotate_1

# echo -- End Rotation 1 --

# echo -- Start Rotation 2 --

# echo 1. original on/off
# python src/gridsearch.py on_off --features features/cis-pd.training.csv \
#                                 --features features/cis-pd.training.rotate_2.csv \
#                                 --labels ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv \
#                                 --filename original_rotate_2

# echo 2. original tremor
# python src/gridsearch.py tremor --features features/cis-pd.training.csv \
#                                 --features features/cis-pd.training.rotate_2.csv \
#                                 --labels ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv \
#                                 --filename original_rotate_2

# echo 3. original dysk
# python src/gridsearch.py dyskinesia --features features/cis-pd.training.csv \
#                                     --features features/cis-pd.training.rotate_2.csv \
#                                     --labels ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv \
#                                     --filename original_rotate_2

# echo -- End Rotation 2 --

# echo -- Start Rotation 3 --

# echo 1. original on/off
# python src/gridsearch.py on_off --features features/cis-pd.training.csv \
#                                 --features features/cis-pd.training.rotate_1.csv \
#                                 --features features/cis-pd.training.rotate_2.csv \
#                                 --labels ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv \
#                                 --filename original_rotate_1_and_2

# echo 2. original tremor
# python src/gridsearch.py tremor --features features/cis-pd.training.csv \
#                                 --features features/cis-pd.training.rotate_1.csv \
#                                 --features features/cis-pd.training.rotate_2.csv \
#                                 --labels ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv \
#                                 --filename original_rotate_1_and_2

# echo 3. original dysk
# python src/gridsearch.py dyskinesia --features features/cis-pd.training.csv \
#                                     --features features/cis-pd.training.rotate_1.csv \
#                                     --features features/cis-pd.training.rotate_2.csv \
#                                     --labels ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv \
#                                     --filename original_rotate_1_and_2

# echo -- End Rotation 3 --

# echo -- Start Rotation combhpfnoinact 1 --

# echo 1. combhpfnoinact on/off
# python src/gridsearch.py on_off --features features/cis-pd.training.combhpfnoinact.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.rotate_1.csv \
#                                 --labels ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv \
#                                 --filename combhpfnoinact_rotate_1

# echo 2. combhpfnoinact tremor
# python src/gridsearch.py tremor --features features/cis-pd.training.combhpfnoinact.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.rotate_1.csv \
#                                 --labels ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv \
#                                 --filename combhpfnoinact_rotate_1 \
#                                 --pred_path ${msek_path} \
#                                 --msek_path ${msek_path}

# echo 3. combhpfnoinact dysk
# python src/gridsearch.py dyskinesia --features features/cis-pd.training.combhpfnoinact.csv \
#                                     --features features/cis-pd.training.combhpfnoinact.rotate_1.csv \
#                                     --labels ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv \
#                                     --filename combhpfnoinact_rotate_1 \
#                                     --pred_path ${msek_path} \
#                                     --msek_path ${msek_path}

# echo -- End Rotation combhpfnoinact 1 --

# echo -- Start Rotation combhpfnoinact 2 --

# echo 1. combhpfnoinact on/off
# python src/gridsearch.py on_off --features features/cis-pd.training.combhpfnoinact.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.rotate_2.csv \
#                                 --labels ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv \
#                                 --filename combhpfnoinact_rotate_2

# echo 2. combhpfnoinact tremor
# python src/gridsearch.py tremor --features features/cis-pd.training.combhpfnoinact.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.rotate_2.csv \
#                                 --labels ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv \
#                                 --filename combhpfnoinact_rotate_2

# echo 3. combhpfnoinact dysk
# python src/gridsearch.py dyskinesia --features features/cis-pd.training.combhpfnoinact.csv \
#                                     --features features/cis-pd.training.combhpfnoinact.rotate_2.csv \
#                                     --labels ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv \
#                                     --filename combhpfnoinact_rotate_2

# echo -- End Rotation combhpfnoinact 2 --

# echo -- Start Rotation combhpfnoinact 3 --

# echo 1. combhpfnoinact on/off
# python src/gridsearch.py on_off --features features/cis-pd.training.combhpfnoinact.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.rotate_1.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.rotate_2.csv \
#                                 --labels ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv \
#                                 --filename combhpfnoinact_rotate_1_and_2

# echo 2. combhpfnoinact tremor
# python src/gridsearch.py tremor --features features/cis-pd.training.combhpfnoinact.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.rotate_1.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.rotate_2.csv \
#                                 --labels ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv \
#                                 --filename combhpfnoinact_rotate_1_and_2

# echo 3. combhpfnoinact dysk
# python src/gridsearch.py dyskinesia --features features/cis-pd.training.combhpfnoinact.csv \
#                                     --features features/cis-pd.training.combhpfnoinact.rotate_1.csv \
#                                     --features features/cis-pd.training.combhpfnoinact.rotate_2.csv \
#                                     --labels ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv \
#                                     --filename combhpfnoinact_rotate_1_and_2

# echo -- End Rotation combhpfnoinact 3 --

# echo ---- Done with noise Data Augmentation ----

# echo ---- Combining Data Augmentation methods ---- 

# echo ---- For tremor, combining combhpfnoinact + combhpfnoinact resample 0.9 + combhpfnoinact rotate 1 --------

# python src/gridsearch.py tremor --features features/cis-pd.training.combhpfnoinact.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.resample_0.9.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.rotate_1.csv \
#                                 --labels ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv \
#                                 --filename combhpfnoinact_resample_0.9_rotate_1


# echo ---- Combination ------

# python src/gridsearch.py on_off --features features/cis-pd.training.combhpfnoinact.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.resample_0.9.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.rotate_1.csv \
#                                 --labels ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv \
#                                 --filename combhpfnoinact.three

# echo 2. original tremor
# python src/gridsearch.py tremor --features features/cis-pd.training.combhpfnoinact.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.noise_mu_0_sig_0.1.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.resample_0.9.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.resample_1.1.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.rotate_1.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.rotate_2.csv \
#                                 --labels ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv \
#                                 --filename combhpfnoinact.combinaison

# python src/gridsearch.py dyskinesia --features features/cis-pd.training.combhpfnoinact.csv \
#                                     --features features/cis-pd.training.combhpfnoinact.resample_0.9.csv \
#                                     --features features/cis-pd.training.combhpfnoinact.rotate_1.csv \
#                                     --labels ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv \
#                                     --filename combhpfnoinact.three


# echo ---- Combine 5 rotations ----- 

# python src/gridsearch.py on_off --features features/cis-pd.training.combhpfnoinact.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.rotate_1.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.rotate_2.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.rotate_3.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.rotate_4.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.rotate_5.csv \
#                                 --labels ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv \
#                                 --pred_path ${msek_path} \
#                                 --msek_path ${msek_path} \
#                                 --filename combhpfnoinact.rotation_comb_5

# echo 2. original tremor
# python src/gridsearch.py tremor --features features/cis-pd.training.combhpfnoinact.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.rotate_1.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.rotate_2.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.rotate_3.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.rotate_4.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.rotate_5.csv \
#                                 --labels ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv \
#                                 --pred_path ${msek_path} \
#                                 --msek_path ${msek_path} \
#                                 --filename combhpfnoinact.rotation_comb_5

# python src/gridsearch.py dyskinesia --features features/cis-pd.training.combhpfnoinact.csv \
#                                     --features features/cis-pd.training.combhpfnoinact.rotate_1.csv \
#                                     --features features/cis-pd.training.combhpfnoinact.rotate_2.csv \
#                                     --features features/cis-pd.training.combhpfnoinact.rotate_3.csv \
#                                     --features features/cis-pd.training.combhpfnoinact.rotate_4.csv \
#                                     --features features/cis-pd.training.combhpfnoinact.rotate_5.csv \
#                                     --labels ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv \
#                                     --pred_path ${msek_path} \
#                                     --msek_path ${msek_path} \
#                                     --filename combhpfnoinact.rotation_comb_5

# echo ---- End Combine 5 rotations ----- 


# echo ---- Combine Rotation Bound and Gaussian Noise ----- 

# python src/gridsearch.py on_off --features features/cis-pd.training.combhpfnoinact.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.rotate_bound_5.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.rotate_bound_10.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.rotate_bound_15.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.rotate_bound_20.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.rotate_bound_25.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.rotate_bound_30.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.noise_mu_0_sig_0.1.csv \
#                                 --labels ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv \
#                                 --pred_path ${msek_path} \
#                                 --msek_path ${msek_path} \
#                                 --filename combhpfnoinact.rotate_bound_noise_incl25

# python src/gridsearch.py tremor --features features/cis-pd.training.combhpfnoinact.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.rotate_bound_5.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.rotate_bound_10.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.rotate_bound_15.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.rotate_bound_20.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.rotate_bound_25.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.rotate_bound_30.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.noise_mu_0_sig_0.1.csv \
#                                 --labels ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv \
#                                 --pred_path ${msek_path} \
#                                 --msek_path ${msek_path} \
#                                 --filename combhpfnoinact.rotate_bound_noise_incl25

# python src/gridsearch.py dyskinesia --features features/cis-pd.training.combhpfnoinact.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.rotate_bound_5.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.rotate_bound_10.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.rotate_bound_15.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.rotate_bound_20.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.rotate_bound_25.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.rotate_bound_30.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.noise_mu_0_sig_0.1.csv \
#                                 --labels ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv \
#                                 --pred_path ${msek_path} \
#                                 --msek_path ${msek_path} \
#                                 --filename combhpfnoinact.rotate_bound_noise_incl25

# echo Start combhpfnoinact.rotate_bound_noise_excl25

# python src/gridsearch.py on_off --features features/cis-pd.training.combhpfnoinact.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.rotate_bound_5.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.rotate_bound_10.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.rotate_bound_15.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.rotate_bound_20.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.rotate_bound_30.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.noise_mu_0_sig_0.1.csv \
#                                 --labels ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv \
#                                 --pred_path ${msek_path} \
#                                 --msek_path ${msek_path} \
#                                 --filename combhpfnoinact.rotate_bound_noise_excl25

# python src/gridsearch.py tremor --features features/cis-pd.training.combhpfnoinact.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.rotate_bound_5.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.rotate_bound_10.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.rotate_bound_15.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.rotate_bound_20.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.rotate_bound_30.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.noise_mu_0_sig_0.1.csv \
#                                 --labels ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv \
#                                 --pred_path ${msek_path} \
#                                 --msek_path ${msek_path} \
#                                 --filename combhpfnoinact.rotate_bound_noise_excl25

# python src/gridsearch.py dyskinesia --features features/cis-pd.training.combhpfnoinact.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.rotate_bound_5.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.rotate_bound_10.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.rotate_bound_15.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.rotate_bound_20.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.rotate_bound_30.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.noise_mu_0_sig_0.1.csv \
#                                 --labels ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv \
#                                 --pred_path ${msek_path} \
#                                 --msek_path ${msek_path} \
#                                 --filename combhpfnoinact.rotate_bound_noise_excl25

# echo End combhpfnoinact.rotate_bound_noise_excl25


# python src/gridsearch.py dyskinesia --features features/cis-pd.training.combhpfnoinact.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.rotate_bound_5.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.rotate_bound_15.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.rotate_bound_30.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.noise_mu_0_sig_0.1.csv \
#                                 --labels ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv \
#                                 --pred_path ${msek_path} \
#                                 --msek_path ${msek_path} \
#                                 --filename combhpfnoinact.rotate_bound_noise_5_15_30

# python src/gridsearch.py on_off --features features/cis-pd.training.combhpfnoinact.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.rotate_5.csv \
#                                 --labels ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv \
#                                 --pred_path ${msek_path} \
#                                 --msek_path ${msek_path} \
#                                 --filename combhpfnoinact.rotate_5

# python src/gridsearch.py tremor --features features/cis-pd.training.combhpfnoinact.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.rotate_1.csv \
#                                 --labels ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv \
#                                 --pred_path ${msek_path} \
#                                 --msek_path ${msek_path} \
#                                 --filename ccombhpfnoinact.rotate_1

# python src/gridsearch.py tremor --features features/cis-pd.training.combhpfnoinact.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.rotate_3.csv \
#                                 --labels ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv \
#                                 --pred_path ${msek_path} \
#                                 --msek_path ${msek_path} \
#                                 --filename ccombhpfnoinact.rotate_3

# python src/gridsearch.py tremor --features features/cis-pd.training.combhpfnoinact.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.rotate_4.csv \
#                                 --labels ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv \
#                                 --pred_path ${msek_path} \
#                                 --msek_path ${msek_path} \
#                                 --filename ccombhpfnoinact.rotate_4

# python src/gridsearch.py tremor --features features/cis-pd.training.combhpfnoinact.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.rotate_5.csv \
#                                 --labels ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv \
#                                 --pred_path ${msek_path} \
#                                 --msek_path ${msek_path} \
#                                 --filename ccombhpfnoinact.rotate_5
                        
# python src/gridsearch.py dyskinesia --features features/cis-pd.training.combhpfnoinact.csv \
#                                 --features features/cis-pd.training.combhpfnoinact.rotate_4.csv \
#                                 --labels ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv \
#                                 --pred_path ${msek_path} \
#                                 --msek_path ${msek_path} \
#                                 --filename ccombhpfnoinact.rotate_4 \
#                                 --lofo_importance yes


echo 2. combhpfnoinact tremor
python src/gridsearch.py tremor --features features/cis-pd.training.combhpfnoinact.csv \
                                --features features/cis-pd.training.combhpfnoinact.resample_0.9.csv \
                                --features features/cis-pd.training.combhpfnoinact.rotate_1.csv \
                                --labels ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv \
                                --pred_path ${msek_path} \
                                --msek_path ${msek_path} \
                                --filename combhpfnoinact.rotate_1_resample09

echo 2. combhpfnoinact tremor
python src/gridsearch.py tremor --features features/cis-pd.training.combhpfnoinact.csv \
                                --features features/cis-pd.training.combhpfnoinact.rotate_2.csv \
                                --features features/cis-pd.training.combhpfnoinact.resample_0.9.csv \
                                --labels ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv \
                                --pred_path ${msek_path} \
                                --msek_path ${msek_path} \
                                --filename combhpfnoinact.rotate_2_resample09

python src/gridsearch.py tremor --features features/cis-pd.training.combhpfnoinact.csv \
                                --features features/cis-pd.training.combhpfnoinact.rotate_3.csv \
                                --features features/cis-pd.training.combhpfnoinact.resample_0.9.csv \
                                --labels ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv \
                                --pred_path ${msek_path} \
                                --msek_path ${msek_path} \
                                --filename combhpfnoinact.rotate_3_resample09

python src/gridsearch.py tremor --features features/cis-pd.training.combhpfnoinact.csv \
                                --features features/cis-pd.training.combhpfnoinact.rotate_4.csv \
                                --features features/cis-pd.training.combhpfnoinact.resample_0.9.csv \
                                --labels ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv \
                                --pred_path ${msek_path} \
                                --msek_path ${msek_path} \
                                --filename combhpfnoinact.rotate_4_resample09

python src/gridsearch.py tremor --features features/cis-pd.training.combhpfnoinact.csv \
                                --features features/cis-pd.training.combhpfnoinact.rotate_5.csv \
                                --features features/cis-pd.training.combhpfnoinact.resample_0.9.csv \
                                --labels ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv \
                                --pred_path ${msek_path} \
                                --msek_path ${msek_path} \
                                --filename combhpfnoinact.rotate_5_resample09