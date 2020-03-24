#!/usr/bin/env bash

chmod 755 /home/sjoshi/codes/python/BeatPD/code/run_auto.sh
if [ -f path.sh ]; then . ./path.sh; fi

eval "$(conda shell.bash hook)"
conda activate BeatPD 

# Path to the features we want to use 
#sDirFeats=/export/b03/sbhati/PD/BeatPD/AE_feats
#sOut=$expDir/ivec_${ivecDim}/resiVecKNN_Fold${foldN}

# Path to where we want to log file about this script 
sOut=$1 
ivecDim=$2
sDirFeats=$3
# Only do the nomber of components that we have the corresponding ivectors
#ivecDim=300

# Only run stage 5 which means only the SVR step 
stage=5

cmd=/export/c08/lmorove1/kaldi/egs/beatPDivec/v1_autoenc/utils/run.pl

for ((iNumFold=0; iNumFold <=4 ; iNumFold++))  do
    $cmd $sOut/pca_${numComponents}_svr_runauto_Testx.log \
    /home/sjoshi/codes/python/BeatPD/code/run_auto.sh $iNumFold $sDirFeats $ivecDim $stage || exit 1;
done

source deactivate
