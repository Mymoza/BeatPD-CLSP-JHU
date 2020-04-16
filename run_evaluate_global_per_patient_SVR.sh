#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate BeatPD

export PATH="$HOME/anaconda3/bin:$PATH"
export PATH="./local:$PATH"

#if [ -f path.sh ]; then . ./path.sh; fi
#. parse_options.sh || exit 1;


for ivecDim in 350 400 450 500 550; do
    sFileTrai=/export/c08/lmorove1/kaldi/egs/beatPDivec/dysk_noinact_auto60_480fl/exp/ivec_$ivecDim/
    sOut=$sFileTrai
    sSubchallenge=dysk

    #filePath=/home/sjoshi/codes/python/BeatPD/code/ 
    filePath=`pwd`/

    ./evaluate_global_per_patient_SVR.sh $sFileTrai $sOut $sSubchallenge
done

conda deactivate
