#!/bin/bash

####################################
# REPRODUCE SUBMISSION 4 FOR REAL-PD
#####################################

conda activate BeatPD_xgboost 

# create features for training and testing
recog_set="phoneacc_test.scp  phoneacc_total.scp watchacc_test.scp  watchacc_total.scp  watchgyr_test.scp  watchgyr_total.scp"
nj=32
logdir=exp
export decode_cmd="utils/queue.pl --mem 4G"
set -e

#for rtask in ${recog_set}; do
#(
#  split_segments=""
#  for n in $(seq $nj); do
#    split_segments="$split_segments $logdir/${rtask}.$n"
#  done

#  utils/split_scp.pl data/${rtask} $split_segments
#  ${decode_cmd} JOB=1:${nj} $logdir/${rtask}.JOB.log \
#    ./submit_realpd.sh $logdir/${rtask}.JOB $logdir/${rtask}.JOB.csv
#) &
#pids+=($!) # store background pids
#done
#i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
#[ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false

# Merge the 32 subsets into one file features/watchgyr_total.csv
#for rtask in ${recog_set}; do
#  awk '{if(NR==1||FNR>1)print;}' $logdir/${rtask}.*.csv > features/${rtask}.csv
#done

#echo "Finished creating features"

echo "Starting Gridsearch"
echo "Warning: Right now you can only save one config at a time, it will rewrite the pickle config file"
echo "Fix: Add an argument with the subtype to save it as subtybe and task" 

path_labels=/home/sjoshi/codes/python/BeatPD/data/BeatPD/real-pd.data_labels/
echo "1. Tremor"
python src/gridsearch_realpd.py tremor features/watchgyr_total.scp.csv ${path_labels}/REAL-PD_Training_Data_IDs_Labels.csv data/watchgyr_order.csv data/watchgyr_fold0.csv data/watchgyr_fold1.csv data/watchgyr_fold2.csv data/watchgyr_fold3.csv data/watchgyr_fold4.csv
#python src/gridsearch_realpd.py tremor features/phoneacc_total.scp.csv ${path_labels}/REAL-PD_Training_Data_IDs_Labels.csv data/phoneacc_order.csv data/phoneacc_fold0.csv data/phoneacc_fold1.csv data/phoneacc_fold2.csv data/phoneacc_fold3.csv data/phoneacc_fold4.csv
#python src/gridsearch_realpd.py tremor features/watchacc_total.scp.csv ${path_labels}/REAL-PD_Training_Data_IDs_Labels.csv data/watchacc_order.csv data/watchacc_fold0.csv data/watchacc_fold1.csv data/watchacc_fold2.csv data/watchacc_fold3.csv data/watchacc_fold4.csv

echo "2. Dyskinesia"
#python src/gridsearch_realpd.py dyskinesia features/watchgyr_total.scp.csv ${path_labels}/REAL-PD_Training_Data_IDs_Labels.csv data/watchgyr_order.csv data/watchgyr_fold0.csv data/watchgyr_fold1.csv data/watchgyr_fold2.csv data/watchgyr_fold3.csv data/watchgyr_fold4.csv
#python src/gridsearch_realpd.py dyskinesia features/phoneacc_total.scp.csv ${path_labels}/REAL-PD_Training_Data_IDs_Labels.csv data/phoneacc_order.csv data/phoneacc_fold0.csv data/phoneacc_fold1.csv data/phoneacc_fold2.csv data/phoneacc_fold3.csv data/phoneacc_fold4.csv
#python src/gridsearch_realpd.py dyskinesia features/watchacc_total.scp.csv ${path_labels}/REAL-PD_Training_Data_IDs_Labels.csv data/watchacc_order.csv data/watchacc_fold0.csv data/watchacc_fold1.csv data/watchacc_fold2.csv data/watchacc_fold3.csv data/watchacc_fold4.csv

echo "3. On_Off"
#python src/gridsearch_realpd.py on_off features/watchgyr_total.scp.csv ${path_labels}/REAL-PD_Training_Data_IDs_Labels.csv data/watchgyr_order.csv data/watchgyr_fold0.csv data/watchgyr_fold1.csv data/watchgyr_fold2.csv data/watchgyr_fold3.csv data/watchgyr_fold4.csv
#python src/gridsearch_realpd.py on_off features/phoneacc_total.scp.csv ${path_labels}/REAL-PD_Training_Data_IDs_Labels.csv data/phoneacc_order.csv data/phoneacc_fold0.csv data/phoneacc_fold1.csv data/phoneacc_fold2.csv data/phoneacc_fold3.csv data/phoneacc_fold4.csv
#python src/gridsearch_realpd.py on_off features/watchacc_total.scp.csv ${path_labels}/REAL-PD_Training_Data_IDs_Labels.csv data/watchacc_order.csv data/watchacc_fold0.csv data/watchacc_fold1.csv data/watchacc_fold2.csv data/watchacc_fold3.csv data/watchacc_fold4.csv

echo "End Of GridSearch"

echo "Start to predict"

# Only predicting on Dyskinesia as that's what we submitted in the final submission 

#echo "1. Tremor"
# predict_realpd.py [tremor, on_off, dyskinesia] [features in csv file for subtype] [Real-PD Training Labels] [features in csv file for test subtype] [Real-PD Test Labels (measurement_id, subjects_id)] [watchgyr, phoneacc, watchacc]

#python src/predict_realpd.py tremor features/watchgyr_total.scp.csv ${path_labels}/REAL-PD_Training_Data_IDs_Labels.csv features/watchgyr_test.scp.csv ${path_labels}/REAL-PD_Test_Data_IDs_Labels.csv data/BEAT-PD_SC3_Tremor_Submission_Template.csv watchgyr

#python src/predict_realpd.py tremor features/phoneacc_total.scp.csv ${path_labels}/REAL-PD_Training_Data_IDs_Labels.csv features/phoneacc_test.scp.csv ${path_labels}/REAL-PD_Test_Data_IDs_Labels.csv data/BEAT-PD_SC3_Tremor_Submission_Template.csv phoneacc

#python src/predict_realpd.py tremor features/watchacc_total.scp.csv ${path_labels}/REAL-PD_Training_Data_IDs_Labels.csv features/watchacc_test.scp.csv ${path_labels}/REAL-PD_Test_Data_IDs_Labels.csv data/BEAT-PD_SC3_Tremor_Submission_Template.csv watchacc

echo "2. Dyskinesia"
python src/predict_realpd.py dyskinesia features/watchgyr_total.scp.csv ${path_labels}/REAL-PD_Training_Data_IDs_Labels.csv features/watchgyr_test.scp.csv ${path_labels}/REAL-PD_Test_Data_IDs_Labels.csv data/BEAT-PD_SC2_Dyskinesia_Submission_Template.csv watchgyr

python src/predict_realpd.py dyskinesia features/phoneacc_total.scp.csv ${path_labels}/REAL-PD_Training_Data_IDs_Labels.csv features/phoneacc_test.scp.csv ${path_labels}/REAL-PD_Test_Data_IDs_Labels.csv data/BEAT-PD_SC2_Dyskinesia_Submission_Template.csv phoneacc

python src/predict_realpd.py dyskinesia features/watchacc_total.scp.csv ${path_labels}/REAL-PD_Training_Data_IDs_Labels.csv features/watchacc_test.scp.csv ${path_labels}/REAL-PD_Test_Data_IDs_Labels.csv data/BEAT-PD_SC2_Dyskinesia_Submission_Template.csv watchacc

#echo "3. On_Off" 
#python src/predict_realpd.py on_off features/watchgyr_total.scp.csv ${path_labels}/REAL-PD_Training_Data_IDs_Labels.csv features/watchgyr_test.scp.csv ${path_labels}/REAL-PD_Test_Data_IDs_Labels.csv data/BEAT-PD_SC1_OnOff_Submission_Template.csv watchgyr

#python src/predict_realpd.py on_off features/phoneacc_total.scp.csv ${path_labels}/REAL-PD_Training_Data_IDs_Labels.csv features/phoneacc_test.scp.csv ${path_labels}/REAL-PD_Test_Data_IDs_Labels.csv data/BEAT-PD_SC1_OnOff_Submission_Template.csv phoneacc

#python src/predict_realpd.py on_off features/watchacc_total.scp.csv ${path_labels}/REAL-PD_Training_Data_IDs_Labels.csv features/watchacc_test.scp.csv ${path_labels}/REAL-PD_Test_Data_IDs_Labels.csv data/BEAT-PD_SC1_OnOff_Submission_Template.csv watchacc
echo "End of predictions"

conda deactivate
echo "End of run_realpd.sh"

