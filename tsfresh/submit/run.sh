#!/bin/bash

####################################
# REPRODUCE SUBMISSION 3 FOR CIS-PD 
#####################################

conda activate BeatPD_xgboost

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

# run grid search
echo "Starting Gridsearch" 
python src/gridsearch.py tremor features/cis-pd.training.csv data/label.csv
echo "End of the gridsarch"

#generate submission files
echo "Generate Submission Files is starting" 
python src/predict.py tremor features/cis-pd.training.csv data/label.csv features/cis-pd.testing.csv /home/sjoshi/codes/python/BeatPD/data/BeatPD/cis-pd.data_labels/CIS-PD_Test_Data_IDs_Labels.csv data/BEAT-PD_SC3_Tremor_Submission_Template.csv
python src/predict.py dyskinesia features/cis-pd.training.csv data/label.csv features/cis-pd.testing.csv /home/sjoshi/codes/python/BeatPD/data/BeatPD/cis-pd.data_labels/CIS-PD_Test_Data_IDs_Labels.csv data/BEAT-PD_SC2_Dyskinesia_Submission_Template.csv
python src/predict.py on_off features/cis-pd.training.csv data/label.csv features/cis-pd.testing.csv /home/sjoshi/codes/python/BeatPD/data/BeatPD/cis-pd.data_labels/CIS-PD_Test_Data_IDs_Labels.csv data/BEAT-PD_SC1_OnOff_Submission_Template.csv

#echo "End of generating submission files"

echo "Section checking if there are differences between your submission files and the 3rd submission of JHU-CLSP"
echo "For CIS-PD 3rd Submission" 
diff -q submission/cis-pd.tremor.csv submission3_preds/cis-pd.tremor.csv 
diff -q submission/cis-pd.dyskinesia.csv submission3_preds/cis-pd.dyskinesia.csv
diff -q submission/cis-pd.on_off.csv submission3_preds/cis-pd.on_off.csv
echo "End of the section checking if there is any difference between your submission files and the 3rd submission of JHU-CLSP"

conda deactivate
echo "run.sh is all done"
