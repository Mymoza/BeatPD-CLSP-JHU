#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate plda2

cmd=run.pl

export PATH="$HOME/anaconda3/bin:$PATH"
export PATH="./local:$PATH"

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

sFileTrai=$1
sOut=$2

cmd=run.pl
$cmd $sOut/globalAccuPLDA.log \
     get_final_scores_accuracy.py  --file-path $sFileTrai \

$cmd $sOut/globalAccuKNN.log \
     get_final_scores_accuracy.py  --file-path $sFileTrai \
     --is-knn
conda deactivate
