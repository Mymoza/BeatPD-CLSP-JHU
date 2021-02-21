#!/bin/bash

###########################################################
# This file is used to get predictions on the test folds
###########################################################

conda activate BeatPD_xgboost

path_labels_cis=/home/sjoshi/codes/python/BeatPD/data/BeatPD/cis-pd.data_labels/
path_labels_real=/home/sjoshi/codes/python/BeatPD/data/BeatPD/real-pd.data_labels/

# echo "TREMOR"
# Everyone has the same parameters? 

# CIS-PD
# python src/gridsearch.py on_off features/cis-pd.training.csv ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv 0
# python src/gridsearch.py tremor features/cis-pd.training.csv ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv 0 
# python src/gridsearch.py dyskinesia features/cis-pd.training.csv ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv 0


python src/gridsearch.py  --features "/home/mpgill/BeatPD/BeatPD-CLSP-JHU/tsfresh/submit/features/cis-pd.training.csv" \
                            --labels "/home/sjoshi/codes/python/BeatPD/data/BeatPD/cis-pd.data_labels/CIS-PD_Training_Data_IDs_Labels.csv" \
                            --pred_path "/home/mpgill/BeatPD/BeatPD-CLSP-JHU/tsfresh/submit/linear_comb/" \
                            --linear_combination 0.2

python src/gridsearch.py tremor  --features "/home/mpgill/BeatPD/BeatPD-CLSP-JHU/tsfresh/submit/features/cis-pd.training.csv" \
                                 --labels "/home/sjoshi/codes/python/BeatPD/data/BeatPD/cis-pd.data_labels/CIS-PD_Training_Data_IDs_Labels.csv" \
                                 --pred_path "/home/mpgill/BeatPD/BeatPD-CLSP-JHU/tsfresh/submit/linear_comb/" \
                                 --linear_combination 0.2

python src/gridsearch.py dyskinesia  --features "/home/mpgill/BeatPD/BeatPD-CLSP-JHU/tsfresh/submit/features/cis-pd.training.csv" \
                                     --labels "/home/sjoshi/codes/python/BeatPD/data/BeatPD/cis-pd.data_labels/CIS-PD_Training_Data_IDs_Labels.csv" \
                                     --pred_path "/home/mpgill/BeatPD/BeatPD-CLSP-JHU/tsfresh/submit/linear_comb/" \
                                    --linear_combination 0.2


python src/gridsearch.py  --features "/home/mpgill/BeatPD/BeatPD-CLSP-JHU/tsfresh/submit/features/cis-pd.training.csv" \
                            --labels "/home/sjoshi/codes/python/BeatPD/data/BeatPD/cis-pd.data_labels/CIS-PD_Training_Data_IDs_Labels.csv" \
                            --pred_path "/home/mpgill/BeatPD/BeatPD-CLSP-JHU/tsfresh/submit/linear_comb/" \
                            --linear_combination 0.4

python src/gridsearch.py tremor  --features "/home/mpgill/BeatPD/BeatPD-CLSP-JHU/tsfresh/submit/features/cis-pd.training.csv" \
                                 --labels "/home/sjoshi/codes/python/BeatPD/data/BeatPD/cis-pd.data_labels/CIS-PD_Training_Data_IDs_Labels.csv" \
                                 --pred_path "/home/mpgill/BeatPD/BeatPD-CLSP-JHU/tsfresh/submit/linear_comb/" \
                                 --linear_combination 0.4

python src/gridsearch.py dyskinesia  --features "/home/mpgill/BeatPD/BeatPD-CLSP-JHU/tsfresh/submit/features/cis-pd.training.csv" \
                                     --labels "/home/sjoshi/codes/python/BeatPD/data/BeatPD/cis-pd.data_labels/CIS-PD_Training_Data_IDs_Labels.csv" \
                                     --pred_path "/home/mpgill/BeatPD/BeatPD-CLSP-JHU/tsfresh/submit/linear_comb/" \
                                    --linear_combination 0.4


python src/gridsearch.py  --features "/home/mpgill/BeatPD/BeatPD-CLSP-JHU/tsfresh/submit/features/cis-pd.training.csv" \
                            --labels "/home/sjoshi/codes/python/BeatPD/data/BeatPD/cis-pd.data_labels/CIS-PD_Training_Data_IDs_Labels.csv" \
                            --pred_path "/home/mpgill/BeatPD/BeatPD-CLSP-JHU/tsfresh/submit/linear_comb/" \
                            --linear_combination 0.001

python src/gridsearch.py tremor  --features "/home/mpgill/BeatPD/BeatPD-CLSP-JHU/tsfresh/submit/features/cis-pd.training.csv" \
                                 --labels "/home/sjoshi/codes/python/BeatPD/data/BeatPD/cis-pd.data_labels/CIS-PD_Training_Data_IDs_Labels.csv" \
                                 --pred_path "/home/mpgill/BeatPD/BeatPD-CLSP-JHU/tsfresh/submit/linear_comb/" \
                                 --linear_combination 0.001

python src/gridsearch.py dyskinesia  --features "/home/mpgill/BeatPD/BeatPD-CLSP-JHU/tsfresh/submit/features/cis-pd.training.csv" \
                                     --labels "/home/sjoshi/codes/python/BeatPD/data/BeatPD/cis-pd.data_labels/CIS-PD_Training_Data_IDs_Labels.csv" \
                                     --pred_path "/home/mpgill/BeatPD/BeatPD-CLSP-JHU/tsfresh/submit/linear_comb/" \
                                    --linear_combination 0.001



# echo "DYSKINESIA"
# python src/getpreds_perpatient.py on_off features/cis-pd.training.csv ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv
# python src/getpreds_perpatient.py tremor features/cis-pd.training.csv ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv
# python src/getpreds_perpatient.py dyskinesia features/cis-pd.training.csv ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv