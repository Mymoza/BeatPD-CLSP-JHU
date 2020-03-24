#!/usr/bin/env bash

chmod 755 /home/sjoshi/codes/python/BeatPD/code/run_auto.sh
if [ -f path.sh ]; then . ./path.sh; fi

eval "$(conda shell.bash hook)"
conda activate BeatPD 

sDirFeats=/export/b03/sbhati/PD/BeatPD/AE_feats
#sOut=$expDir/ivec_${ivecDim}/resiVecKNN_Fold${foldN}

ivecDim=50
stage=4
# FIXME CHANGE THIS FOLD NO 
for ((iNumFold=0; iNumFold <=4 ; iNumFold++))  do
	/home/sjoshi/codes/python/BeatPD/code/run_auto.sh $iNumFold $sDirFeats $ivecDim $stage || exit 1;
done

source deactivate
