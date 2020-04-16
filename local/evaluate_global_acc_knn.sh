#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate BeatPD

export PATH="$HOME/anaconda3/bin:$PATH"
export PATH="./local:$PATH"

#if [ -f path.sh ]; then . ./path.sh; fi
#. parse_options.sh || exit 1;

sFileTrai=$1
sOut=$2

filePath=`pwd`/local/

cmd=/export/c08/lmorove1/kaldi/egs/beatPDivec/v1_autoenc/utils/run.pl
$cmd $sOut/globalAccuPLDA.log \
     ${filePath}get_final_scores_accuracy.py --file-path $sFileTrai \

$cmd $sOut/globalAccuKNN.log \
     ${filePath}get_final_scores_accuracy.py  --file-path $sFileTrai \
     --is-knn
conda deactivate
