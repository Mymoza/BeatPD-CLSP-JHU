#!/bin/bash

###########################################################
# This file reproduces our final submission for Approach 1 - tsfresh + xgboost  
###########################################################

conda activate BeatPD_xgboost 

path_labels_cis=/home/sjoshi/codes/python/BeatPD/data/BeatPD/cis-pd.data_labels/
path_labels_real=/home/sjoshi/codes/python/BeatPD/data/BeatPD/real-pd.data_labels/

############
# CIS-PD 
###########

# echo "1. Creating features for CIS-PD rotate_bound_5"

# # create features for training and testing
# recog_set="cis-pd.training.combhpfnoinact.rotate_bound_5 cis-pd.testing.combhpfnoinact.rotate_bound_5"
# nj=32
# logdir=exp/combhpfnoinact.rotate_bound_5

# mkdir $logdir
# export decode_cmd="utils/queue.pl --mem 4G"
# set -e

# for rtask in ${recog_set}; do
# (
#   split_segments=""
#   for n in $(seq $nj); do
#     split_segments="$split_segments $logdir/${rtask}.$n"
#   done

#   utils/split_scp.pl data/${rtask}.scp $split_segments
#   ${decode_cmd} JOB=1:${nj} $logdir/${rtask}.JOB.log \
#     ./submit.sh $logdir/${rtask}.JOB $logdir/${rtask}.JOB.csv
# ) &
# pids+=($!) # store background pids
# done
# i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
# [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && true 

# for rtask in ${recog_set}; do
#   awk '{if(NR==1||FNR>1)print;}' $logdir/${rtask}.*.csv > features/${rtask}.csv && true 
# done

# echo "2. Done extracting features for rotate_bound_5"


# echo "3. Creating features for CIS-PD rotate_bound_10"

# # create features for training and testing
# recog_set="cis-pd.training.combhpfnoinact.rotate_bound_10 cis-pd.testing.combhpfnoinact.rotate_bound_10"
# nj=32
# logdir=exp/combhpfnoinact.rotate_bound_10

# mkdir $logdir
# export decode_cmd="utils/queue.pl --mem 4G"
# set -e

# for rtask in ${recog_set}; do
# (
#   split_segments=""
#   for n in $(seq $nj); do
#     split_segments="$split_segments $logdir/${rtask}.$n"
#   done

#   utils/split_scp.pl data/${rtask}.scp $split_segments
#   ${decode_cmd} JOB=1:${nj} $logdir/${rtask}.JOB.log \
#     ./submit.sh $logdir/${rtask}.JOB $logdir/${rtask}.JOB.csv
# ) &
# pids+=($!) # store background pids
# done
# i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
# [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && true 

# for rtask in ${recog_set}; do
#   awk '{if(NR==1||FNR>1)print;}' $logdir/${rtask}.*.csv > features/${rtask}.csv && true 
# done

# echo "4. Done extracting features for rotate_bound_10"

# echo "5. Creating features for CIS-PD rotate_bound_15"

# # create features for training and testing
# recog_set="cis-pd.training.combhpfnoinact.rotate_bound_15 cis-pd.testing.combhpfnoinact.rotate_bound_15"
# nj=32
# logdir=exp/combhpfnoinact.rotate_bound_15

# mkdir $logdir
# export decode_cmd="utils/queue.pl --mem 4G"
# set -e

# for rtask in ${recog_set}; do
# (
#   split_segments=""
#   for n in $(seq $nj); do
#     split_segments="$split_segments $logdir/${rtask}.$n"
#   done

#   utils/split_scp.pl data/${rtask}.scp $split_segments
#   ${decode_cmd} JOB=1:${nj} $logdir/${rtask}.JOB.log \
#     ./submit.sh $logdir/${rtask}.JOB $logdir/${rtask}.JOB.csv
# ) &
# pids+=($!) # store background pids
# done
# i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
# [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && true 

# for rtask in ${recog_set}; do
#   awk '{if(NR==1||FNR>1)print;}' $logdir/${rtask}.*.csv > features/${rtask}.csv && true 
# done

# echo "6. Done extracting features for rotate_bound_15"

# echo "7. Creating features for CIS-PD rotate_bound_20"

# # create features for training and testing
# recog_set="cis-pd.training.combhpfnoinact.rotate_bound_20 cis-pd.testing.combhpfnoinact.rotate_bound_20"
# nj=32
# logdir=exp/combhpfnoinact.rotate_bound_20

# mkdir $logdir
# export decode_cmd="utils/queue.pl --mem 4G"
# set -e

# for rtask in ${recog_set}; do
# (
#   split_segments=""
#   for n in $(seq $nj); do
#     split_segments="$split_segments $logdir/${rtask}.$n"
#   done

#   utils/split_scp.pl data/${rtask}.scp $split_segments
#   ${decode_cmd} JOB=1:${nj} $logdir/${rtask}.JOB.log \
#     ./submit.sh $logdir/${rtask}.JOB $logdir/${rtask}.JOB.csv
# ) &
# pids+=($!) # store background pids
# done
# i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
# [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && true 

# for rtask in ${recog_set}; do
#   awk '{if(NR==1||FNR>1)print;}' $logdir/${rtask}.*.csv > features/${rtask}.csv && true 
# done

# echo "8. Done extracting features for rotate_bound_20"

# echo "9. Creating features for CIS-PD rotate_bound_25"

# # create features for training and testing
# recog_set="cis-pd.training.combhpfnoinact.rotate_bound_25 cis-pd.testing.combhpfnoinact.rotate_bound_25"
# nj=32
# logdir=exp/combhpfnoinact.rotate_bound_25

# mkdir $logdir
# export decode_cmd="utils/queue.pl --mem 4G"
# set -e

# for rtask in ${recog_set}; do
# (
#   split_segments=""
#   for n in $(seq $nj); do
#     split_segments="$split_segments $logdir/${rtask}.$n"
#   done

#   utils/split_scp.pl data/${rtask}.scp $split_segments
#   ${decode_cmd} JOB=1:${nj} $logdir/${rtask}.JOB.log \
#     ./submit.sh $logdir/${rtask}.JOB $logdir/${rtask}.JOB.csv
# ) &
# pids+=($!) # store background pids
# done
# i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
# [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && true 

# for rtask in ${recog_set}; do
#   awk '{if(NR==1||FNR>1)print;}' $logdir/${rtask}.*.csv > features/${rtask}.csv && true 
# done

# echo "10. Done extracting features for rotate_bound_25"

# echo "11. Creating features for CIS-PD rotate_bound_30"

# # create features for training and testing
# recog_set="cis-pd.training.combhpfnoinact.rotate_bound_30 cis-pd.testing.combhpfnoinact.rotate_bound_30"
# nj=32
# logdir=exp/combhpfnoinact.rotate_bound_30

# mkdir $logdir
# export decode_cmd="utils/queue.pl --mem 4G"
# set -e

# for rtask in ${recog_set}; do
# (
#   split_segments=""
#   for n in $(seq $nj); do
#     split_segments="$split_segments $logdir/${rtask}.$n"
#   done

#   utils/split_scp.pl data/${rtask}.scp $split_segments
#   ${decode_cmd} JOB=1:${nj} $logdir/${rtask}.JOB.log \
#     ./submit.sh $logdir/${rtask}.JOB $logdir/${rtask}.JOB.csv
# ) &
# pids+=($!) # store background pids
# done
# i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
# [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && true 

# for rtask in ${recog_set}; do
#   awk '{if(NR==1||FNR>1)print;}' $logdir/${rtask}.*.csv > features/${rtask}.csv && true 
# done

# echo "12. Done extracting features for rotate_bound_30"

# echo "13. Creating features for CIS-PD rotate_bound_35"

# # create features for training and testing
# recog_set="cis-pd.training.combhpfnoinact.rotate_bound_35 cis-pd.testing.combhpfnoinact.rotate_bound_35"
# nj=32
# logdir=exp/combhpfnoinact.rotate_bound_35

# mkdir $logdir
# export decode_cmd="utils/queue.pl --mem 4G"
# set -e

# for rtask in ${recog_set}; do
# (
#   split_segments=""
#   for n in $(seq $nj); do
#     split_segments="$split_segments $logdir/${rtask}.$n"
#   done

#   utils/split_scp.pl data/${rtask}.scp $split_segments
#   ${decode_cmd} JOB=1:${nj} $logdir/${rtask}.JOB.log \
#     ./submit.sh $logdir/${rtask}.JOB $logdir/${rtask}.JOB.csv
# ) &
# pids+=($!) # store background pids
# done
# i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
# [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && true 

# for rtask in ${recog_set}; do
#   awk '{if(NR==1||FNR>1)print;}' $logdir/${rtask}.*.csv > features/${rtask}.csv && true 
# done

# echo "14. Done extracting features for rotate_bound_35"

# echo "15. Creating features for CIS-PD rotate_bound_40"

# # create features for training and testing
# recog_set="cis-pd.training.combhpfnoinact.rotate_bound_40 cis-pd.testing.combhpfnoinact.rotate_bound_40"
# nj=32
# logdir=exp/combhpfnoinact.rotate_bound_40

# mkdir $logdir
# export decode_cmd="utils/queue.pl --mem 4G"
# set -e

# for rtask in ${recog_set}; do
# (
#   split_segments=""
#   for n in $(seq $nj); do
#     split_segments="$split_segments $logdir/${rtask}.$n"
#   done

#   utils/split_scp.pl data/${rtask}.scp $split_segments
#   ${decode_cmd} JOB=1:${nj} $logdir/${rtask}.JOB.log \
#     ./submit.sh $logdir/${rtask}.JOB $logdir/${rtask}.JOB.csv
# ) &
# pids+=($!) # store background pids
# done
# i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
# [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && true 

# for rtask in ${recog_set}; do
#   awk '{if(NR==1||FNR>1)print;}' $logdir/${rtask}.*.csv > features/${rtask}.csv && true 
# done

# echo "16. Done extracting features for rotate_bound_40"






# echo "19. Creating Preds Files now" 


mkdir xgb_rotate_bound
msek_path="$(pwd)/xgb_rotate_bound/"

for repeat in 4 5
do
  for bound in 5 10 15 20 25 30 35 40 45
  do 
    echo "Creating scp file"
    ./create_scp_files.sh combhpfnoinact.rotate_bound_${bound}_${repeat}
    echo "Done creating scp file"

    echo "Creating features for CIS-PD rotate_bound_${bound}_${repeat}"

    # create features for training and testing
    recog_set="cis-pd.training.combhpfnoinact.rotate_bound_${bound}_${repeat} cis-pd.testing.combhpfnoinact.rotate_bound_${bound}_${repeat}"
    nj=32
    logdir=exp/combhpfnoinact.rotate_bound_${bound}_${repeat}

    mkdir $logdir
    export decode_cmd="utils/queue.pl --mem 4G"
    set -e

    for rtask in ${recog_set}; do
    (
      split_segments=""
      for n in $(seq $nj); do
        split_segments="$split_segments $logdir/${rtask}.$n"
      done

      utils/split_scp.pl data/${rtask}.scp $split_segments
      ${decode_cmd} JOB=1:${nj} $logdir/${rtask}.JOB.log \
        ./submit.sh $logdir/${rtask}.JOB $logdir/${rtask}.JOB.csv
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && true 

    for rtask in ${recog_set}; do
      awk '{if(NR==1||FNR>1)print;}' $logdir/${rtask}.*.csv > features/${rtask}.csv && true 
    done

    echo "Done extracting features for rotate_bound_${bound}_${repeat}"

    echo ---- Start predicting pour bound_${bound}_${repeat} ----- 

    python src/gridsearch.py on_off --features features/cis-pd.training.combhpfnoinact.csv \
                                    --features features/cis-pd.training.combhpfnoinact.rotate_bound_${bound}_${repeat}.csv \
                                    --labels ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv \
                                    --pred_path ${msek_path} \
                                    --filename combhpfnoinact.rotate_bound_${bound}_${repeat}

    echo 2. original tremor
    python src/gridsearch.py tremor --features features/cis-pd.training.combhpfnoinact.csv \
                                    --features features/cis-pd.training.combhpfnoinact.rotate_bound_${bound}_${repeat}.csv \
                                    --labels ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv \
                                    --pred_path ${msek_path} \
                                    --filename combhpfnoinact.rotate_bound_${bound}_${repeat}

    python src/gridsearch.py dyskinesia --features features/cis-pd.training.combhpfnoinact.csv \
                                    --features features/cis-pd.training.combhpfnoinact.rotate_bound_${bound}_${repeat}.csv \
                                    --labels ${path_labels_cis}/CIS-PD_Training_Data_IDs_Labels.csv \
                                    --pred_path ${msek_path} \
                                    --filename combhpfnoinact.rotate_bound_${bound}_${repeat}

    echo ---- End rotate_bound_${bound}_${repeat} ----- 

  done
done
