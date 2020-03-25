#!/usr/bin/env bash
# This file can be used to create the pickle files needed to evaluate a backend. 

# sOut /export/c08/lmorove1/kaldi/egs/beatPDivec/v1_autoenc/exp/ 
# ivecDim 50 100 150 200 250 300 
# sDirFeats /export/b03/sbhati/PD/BeatPD/AE_feats

echo Working on on/off 

#sOut=/export/c08/lmorove1/kaldi/egs/beatPDivec/v1_autoenc/exp/
#sDirFeats=/export/c08/lmorove1/kaldi/egs/beatPDivec/v1_autoenc

sOut=/export/c08/lmorove1/kaldi/egs/beatPDivec/v1_3ax/exp/
sDirFeats=/export/c08/lmorove1/kaldi/egs/beatPDivec/v1_3ax

for ivecDim in 50 100 150 200 250 300; do
    echo Working on ${ivecDim}
    ./runSVRFold.sh ${sOut} $ivecDim $sDirFeats
done

sOut=/export/c08/lmorove1/kaldi/egs/beatPDivec/v1_3ax_10mfcc_dysk/exp/
sDirFeats=/export/c08/lmorove1/kaldi/egs/beatPDivec/v1_3ax_10mfcc_dysk

echo Working on dyskenisia 

for ivecDim in 50 100 150 200 250 300; do
    echo Working on ${ivecDim}
    ./runSVRFold.sh ${sOut} $ivecDim $sDirFeats
done

echo Working on tremor 

sOut=/export/c08/lmorove1/kaldi/egs/beatPDivec/v1_3ax_10mfcc_tr/exp/
sDirFeats=/export/c08/lmorove1/kaldi/egs/beatPDivec/v1_3ax_10mfcc_tr

for ivecDim in 50 100 150 200 250 300; do
    echo Working on ${ivecDim}
    ./runSVRFold.sh ${sOut} $ivecDim $sDirFeats
done
