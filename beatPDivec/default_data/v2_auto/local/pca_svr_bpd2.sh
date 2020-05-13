#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate BeatPD

export PATH="$HOME/anaconda3/bin:$PATH"
export PATH="./local:$PATH"

sFileTrai=$1
sFileTest=$2
sOut=$3
iComponents=$4
sKernel=$5
fCValue=$6
fEpsilon=$7

# Just to get C in scientific notation for the name of the log files 
fCValueStr=$(printf "%g" $fCValue)

cmd=utils/run.pl
$cmd $sOut/pca_${iComponents}_svr_${sKernel}_${fCValueStr}_${fEpsilon}Testx.log \
     pca_knn_bpd2.py --input-trai $sFileTrai \
     --input-test $sFileTest \
     --output-file $sOut \
     --iComponents $iComponents \
     --sKernel $sKernel \
     --fCValue $fCValue \
     --fEpsilon $fEpsilon \
     --bPatientPredictionsPkl
conda deactivate
