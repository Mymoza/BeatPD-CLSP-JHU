#!/bin/bash

####################################
# REPRODUCE SUBMISSION 3 FOR CIS-PD AND REAL-PD 
#####################################

conda activate BeatPD_xgboost

# create features for training and testing
# recog_set="cis-pd.training cis-pd.testing"
# nj=32
# logdir=exp

# mkdir $logdir
# export decode_cmd="utils/queue.pl --mem 4G"
# set -e

#for rtask in ${recog_set}; do
#(
#  split_segments=""
#  for n in $(seq $nj); do
#    split_segments="$split_segments $logdir/${rtask}.$n"
#  done

#  utils/split_scp.pl data/${rtask}.scp $split_segments
#  ${decode_cmd} JOB=1:${nj} $logdir/${rtask}.JOB.log \
#    ./submit.sh $logdir/${rtask}.JOB $logdir/${rtask}.JOB.csv
#) &
#pids+=($!) # store background pids
#done
#i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
#[ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
#
#for rtask in ${recog_set}; do
#  awk '{if(NR==1||FNR>1)print;}' $logdir/${rtask}.*.csv > features/${rtask}.csv
#done
echo "Finished"

path_labels=/home/sjoshi/codes/python/BeatPD/data/BeatPD/cis-pd.data_labels/
path_labels_real=/home/sjoshi/codes/python/BeatPD/data/BeatPD/real-pd.data_labels/

mkdir xgb_weitd_thing
msek_path="$(pwd)/xgb_weitd_thing/"

# run grid search
echo "Starting Gridsearch" 
# python src/gridsearch_weirdthing.py on_off --features features/cis-pd.training.csv --labels /home/sjoshi/codes/python/BeatPD/data/BeatPD/cis-pd.data_labels/CIS-PD_Training_Data_IDs_Labels.csv --pred_path /home/mpgill/BeatPD/BeatPD-CLSP-JHU/tsfresh/submit/xgb_weitd_thing/ --filename weirdthing

python src/gridsearch_weirdthing.py on_off --features features/cis-pd.training.csv \
                                --labels ${path_labels}/CIS-PD_Training_Data_IDs_Labels.csv \
                                --pred_path ${msek_path} \
                                --filename weirdthing_add_avg

python src/gridsearch_weirdthing.py tremor --features features/cis-pd.training.csv \
                                --labels ${path_labels}/CIS-PD_Training_Data_IDs_Labels.csv \
                                --pred_path ${msek_path} \
                                --filename weirdthing_add_avg

python src/gridsearch_weirdthing.py dyskinesia --features features/cis-pd.training.csv \
                                --labels ${path_labels}/CIS-PD_Training_Data_IDs_Labels.csv \
                                --pred_path ${msek_path} \
                                --filename weirdthing_add_avg

python src/gridsearch_weirdthing.py on_off --features features/cis-pd.training.combhpfnoinact.csv \
                                --labels ${path_labels}/CIS-PD_Training_Data_IDs_Labels.csv \
                                --pred_path ${msek_path} \
                                --filename combhpfnoinact.weirdthing_add_avg

python src/gridsearch_weirdthing.py tremor --features features/cis-pd.training.combhpfnoinact.csv \
                                --labels ${path_labels}/CIS-PD_Training_Data_IDs_Labels.csv \
                                --pred_path ${msek_path} \
                                --filename combhpfnoinact.weirdthing_add_avg

python src/gridsearch_weirdthing.py dyskinesia --features features/cis-pd.training.combhpfnoinact.csv \
                                --labels ${path_labels}/CIS-PD_Training_Data_IDs_Labels.csv \
                                --pred_path ${msek_path} \
                                --filename combhpfnoinact.weirdthing_add_avg

echo "End of the gridsarch"

#generate submission files
# echo "Generate Submission Files is starting" 
# python src/predict.py tremor features/cis-pd.training.csv ${path_labels}/CIS-PD_Training_Data_IDs_Labels.csv features/cis-pd.testing.csv ${path_labels}/CIS-PD_Test_Data_IDs_Labels.csv data/BEAT-PD_SC3_Tremor_Submission_Template.csv

#python src/predict.py dyskinesia features/cis-pd.training.csv ${path_labels}/CIS-PD_Training_Data_IDs_Labels.csv features/cis-pd.testing.csv ${path_labels}/CIS-PD_Test_Data_IDs_Labels.csv data/BEAT-PD_SC2_Dyskinesia_Submission_Template.csv

#python src/predict.py on_off features/cis-pd.training.csv ${path_labels}/CIS-PD_Training_Data_IDs_Labels.csv features/cis-pd.testing.csv ${path_labels}/CIS-PD_Test_Data_IDs_Labels.csv data/BEAT-PD_SC1_OnOff_Submission_Template.csv

# echo "End of generating submission files"



################################
# REAL-PD 
###############################

# echo "REAL-PD" 

# echo "Starting to create features"

# # create features for training and testing
# recog_set="phoneacc_test.scp  phoneacc_total.scp watchacc_test.scp  watchacc_total.scp  watchgyr_test.scp  watchgyr_total.scp"
# nj=32
# logdir=exp
# export decode_cmd="utils/queue.pl --mem 4G"
# set -e

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
#
## Merge the 32 subsets into one file features/watchgyr_total.csv
#for rtask in ${recog_set}; do
#  awk '{if(NR==1||FNR>1)print;}' $logdir/${rtask}.*.csv > features/${rtask}.csv
#done

#echo "Finished creating features"

 
# echo "Start to predict"

# echo "Tremor"
# # predict_realpd.py [tremor, on_off, dyskinesia] [features in csv file for subtype] [Real-PD Training Labels] [features in csv file for test subtype] [Real-PD Test Labels (measurement_id, subjects_id)] [watchgyr, phoneacc, watchacc]

# python src/predict_realpd.py tremor features/watchgyr_total.scp.csv ${path_labels_real}/REAL-PD_Training_Data_IDs_Labels.csv features/watchgyr_test.scp.csv ${path_labels_real}/REAL-PD_Test_Data_IDs_Labels.csv data/BEAT-PD_SC3_Tremor_Submission_Template.csv watchgyr

# python src/predict_realpd.py tremor features/phoneacc_total.scp.csv ${path_labels_real}/REAL-PD_Training_Data_IDs_Labels.csv features/phoneacc_test.scp.csv ${path_labels_real}/REAL-PD_Test_Data_IDs_Labels.csv data/BEAT-PD_SC3_Tremor_Submission_Template.csv phoneacc

# python src/predict_realpd.py tremor features/watchacc_total.scp.csv ${path_labels_real}/REAL-PD_Training_Data_IDs_Labels.csv features/watchacc_test.scp.csv ${path_labels_real}/REAL-PD_Test_Data_IDs_Labels.csv data/BEAT-PD_SC3_Tremor_Submission_Template.csv watchacc

# echo "End Tremor"

# conda deactivate
echo "run.sh is all done"
