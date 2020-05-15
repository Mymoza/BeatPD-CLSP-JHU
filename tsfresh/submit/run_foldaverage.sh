#!/bin/bash

###########################################################
# This file reproduces our final submission for ON/OFF 
###########################################################

conda activate BeatPD_xgboost 

############
# CIS-PD 
###########

# create features for training and testing
recog_set="cis-pd.training cis-pd.testing"
nj=32
logdir=exp

#mkdir $logdir
#export decode_cmd="utils/queue.pl --mem 4G"
#set -e

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

#for rtask in ${recog_set}; do
#  awk '{if(NR==1||FNR>1)print;}' $logdir/${rtask}.*.csv > features/${rtask}.csv
#done
#echo "Finished"

path_labels_cis=/home/sjoshi/codes/python/BeatPD/data/BeatPD/cis-pd.data_labels/

python src/foldaverage.py on_off features/cis-pd.training.csv ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv features/cis-pd.testing.csv ${path_labels_cis}/CIS-PD_Test_Data_IDs_Labels.csv data/BEAT-PD_SC1_OnOff_Submission_Template.csv

###########
# REAL-PD
##########

# create features for training and testing
recog_set="phoneacc_test.scp  phoneacc_total.scp watchacc_test.scp  watchacc_total.scp  watchgyr_test.scp  watchgyr_total.scp"
nj=32
logdir=exp
path_labels_real=/home/sjoshi/codes/python/BeatPD/data/BeatPD/real-pd.data_labels/
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

echo "Finished creating features"

python src/foldaverage_realpd.py on_off features/watchgyr_total.scp.csv ${path_labels_real}/REAL-PD_Training_Data_IDs_Labels.csv data/watchgyr_order.csv features/watchgyr_test.scp.csv ${path_labels_real}/REAL-PD_Test_Data_IDs_Labels.csv data/BEAT-PD_SC1_OnOff_Submission_Template.csv watchgyr

python src/foldaverage_realpd.py on_off features/watchacc_total.scp.csv ${path_labels_real}/REAL-PD_Training_Data_IDs_Labels.csv data/watchacc_order.csv features/watchacc_test.scp.csv ${path_labels_real}/REAL-PD_Test_Data_IDs_Labels.csv data/BEAT-PD_SC1_OnOff_Submission_Template.csv watchacc

python src/foldaverage_realpd.py on_off features/phoneacc_total.scp.csv ${path_labels_real}/REAL-PD_Training_Data_IDs_Labels.csv data/phoneacc_order.csv features/phoneacc_test.scp.csv ${path_labels_real}/REAL-PD_Test_Data_IDs_Labels.csv data/BEAT-PD_SC1_OnOff_Submission_Template.csv phoneacc

conda deactivate
