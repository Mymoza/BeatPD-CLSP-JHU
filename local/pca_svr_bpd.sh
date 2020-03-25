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

echo $fEpsilon
echo sssssssssssss
echo $sOut
echo sssssssssssssss

filePath=/home/sjoshi/codes/python/BeatPD/code/

cmd=/export/c08/lmorove1/kaldi/egs/beatPDivec/v1_autoenc/utils/run.pl
$cmd $sOut/pca_${iComponents}_svr_${sKernel}_${fCValue}_${fEpsilon}Testx.log \
     ${filePath}pca_knn_bpd.py --input-trai $sFileTrai \
     --input-test $sFileTest \
     --output-file $sOut \
     --iComponents $iComponents \
     --sKernel $sKernel \
     --fCValue $fCValue \
     --fEpsilon $fEpsilon
conda deactivate
