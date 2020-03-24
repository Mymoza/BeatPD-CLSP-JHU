#!/usr/bin/env bash

# sOut /export/c08/lmorove1/kaldi/egs/beatPDivec/v1_autoenc/exp/ 
# ivecDim 50 100 150 200 250 300 
# sDirFeats /export/b03/sbhati/PD/BeatPD/AE_feats


sOut=/export/c08/lmorove1/kaldi/egs/beatPDivec/v1_dysk_auto/exp/
sDirFeats=/export/c08/lmorove1/kaldi/egs/beatPDivec/v1_dysk_auto

echo Working on dyskenisia 

for ivecDim in 50 100 150 200 250 300; do
    echo Working on ${ivecDim}
    ./runSVRFold.sh ${sOut} $ivecDim $sDirFeats
done

echo Working on tremor 

sOut=/export/c08/lmorove1/kaldi/egs/beatPDivec/v1_trem_auto/exp/
sDirFeats=/export/c08/lmorove1/kaldi/egs/beatPDivec/v1_trem_auto

for ivecDim in 50 100 150 200 250 300; do
    echo Working on ${ivecDim}
    ./runSVRFold.sh ${sOut} $ivecDim $sDirFeats
done
