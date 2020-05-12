#!/usr/bin/env bash

chmod 755 /home/sjoshi/codes/python/BeatPD/code/run_auto.sh
if [ -f path.sh ]; then . ./path.sh; fi

eval "$(conda shell.bash hook)"
conda activate BeatPD 


sOut=$1 
ivecDim=$2
sDirFeats=$3

# Stage 4 is KNN only 
stage=4

for ((iNumFold=0; iNumFold <=4 ; iNumFold++))  do
	/home/sjoshi/codes/python/BeatPD/code/run_auto.sh $iNumFold $sDirFeats $ivecDim $stage || exit 1;
done

source deactivate
