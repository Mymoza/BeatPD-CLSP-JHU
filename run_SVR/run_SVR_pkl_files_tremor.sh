#!/usr/bin/env bash

sOut=/export/c08/lmorove1/kaldi/egs/beatPDivec/v1_trem_auto/exp/
sDirFeats=/export/c08/lmorove1/kaldi/egs/beatPDivec/v1_trem_auto

#sOut=/export/c08/lmorove1/kaldi/egs/beatPDivec/v1_3ax_10mfcc_tr/exp/
#sDirFeats=/export/c08/lmorove1/kaldi/egs/beatPDivec/v1_3ax_10mfcc_tr

for ivecDim in 350; do
    echo Working on ${ivecDim}
    ./runSVRFold.sh ${sOut} $ivecDim $sDirFeats
done
