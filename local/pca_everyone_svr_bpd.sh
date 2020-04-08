#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate BeatPD

export PATH="$HOME/anaconda3/bin:$PATH"
export PATH="./local:$PATH"
echo "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
sFileTrai=$1
sFileTest=$2
sOut=$3
iComponents=$4
sKernel=$5
fCValue=$6
fEpsilon=$7

# Just to get C in scientific notation for the name of the log files 
fCValueStr=$(printf "%g" $fCValue)

filePath=/home/mpgill/BeatPD/BeatPD-CLSP-JHU/

cmd=/export/c08/lmorove1/kaldi/egs/beatPDivec/v1_autoenc/utils/run.pl
$cmd $sOut/pca_${iComponents}_everyone_svr_${sKernel}_${fCValueStr}_${fEpsilon}Testx.log \
     ${filePath}pca_everyone_svr_bpd.py --input-trai $sFileTrai \
     --input-test $sFileTest \
     --output-file $sOut \
     --iComponents $iComponents \
     --sKernel $sKernel \
     --fCValue $fCValue \
     --fEpsilon $fEpsilon
conda deactivate
