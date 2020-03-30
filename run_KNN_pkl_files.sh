#!/usr/bin/env bash
# This file can be used to create the pickle files needed to evaluate a backend.

# sOut /export/c08/lmorove1/kaldi/egs/beatPDivec/v1_autoenc/exp/
# ivecDim 50 100 150 200 250 300
# sDirFeats /export/b03/sbhati/PD/BeatPD/AE_feats

echo Working on tremorf

#sOut=/export/c08/lmorove1/kaldi/egs/beatPDivec/v1_autoenc/exp/
#sDirFeats=/export/c08/lmorove1/kaldi/egs/beatPDivec/v1_autoenc

#sOut=/export/c08/lmorove1/kaldi/egs/beatPDivec/on_off_noinact_auto30/exp/
#sDirFeats=/export/c08/lmorove1/kaldi/egs/beatPDivec/on_off_noinact_auto30

sOut=/export/c08/lmorove1/kaldi/egs/beatPDivec/dysk_noinact_auto30/exp/
sDirFeats=/export/c08/lmorove1/kaldi/egs/beatPDivec/dysk_noinact_auto30

#sOut=/export/c08/lmorove1/kaldi/egs/beatPDivec/trem_noinact_auto30/exp/
#sDirFeats=/export/c08/lmorove1/kaldi/egs/beatPDivec/trem_noinact_auto30

for ivecDim in 450 500; do
    echo Working on ${ivecDim}
    ./runKNNFold.sh ${sOut} $ivecDim $sDirFeats
done

echo DONE 
