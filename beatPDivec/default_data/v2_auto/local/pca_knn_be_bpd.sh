#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate plda2

export PATH="$HOME/anaconda3/bin:$PATH"
export PATH="./local:$PATH"

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

sFileTrai=$1
sFileTest=$2
sOut=$3
iComponents=$4
iNeighbors=$5

cmd=utils/run.pl
$cmd $sOut/pca_${iComponents}_knn_${iNeighbors}Testx.log \
     pca_knn_bpd2.py --input-trai $sFileTrai \
     --input-test $sFileTest \
     --output-file $sOut \
     --iComponents $iComponents \
     --iNeighbors $iNeighbors
conda deactivate
