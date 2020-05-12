#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate BeatPD

export PATH="$HOME/anaconda3/bin:$PATH"
export PATH="./local:$PATH"

#if [ -f path.sh ]; then . ./path.sh; fi
#. parse_options.sh || exit 1;

sFileTrai=$1
sOut=$2
sSubchallenge=$3 

#filePath=/home/sjoshi/codes/python/BeatPD/code/
filePath=`pwd`/local/

cmd=utils/run.pl
$cmd $sOut/globalAccuPerPatientSVR_Test.log \
         ${filePath}get_final_scores_accuracy.py  --file-path $sFileTrai \
         --is-svr --per-subject-svr --database CIS --subchallenge $sSubchallenge

conda deactivate
