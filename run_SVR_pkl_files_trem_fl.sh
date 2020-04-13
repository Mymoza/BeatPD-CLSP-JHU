#!/usr/bin/env bash
# This file can be used to create the pickle files needed to evaluate a backend. 

# sOut /export/c08/lmorove1/kaldi/egs/beatPDivec/v1_autoenc/exp/ 
# ivecDim 50 100 150 200 250 300 
# sDirFeats /export/b03/sbhati/PD/BeatPD/AE_feats

echo Working on trem trem_noinact_auto30_240fl

#sOut=/export/c08/lmorove1/kaldi/egs/beatPDivec/v1_autoenc/exp/
#sDirFeats=/export/c08/lmorove1/kaldi/egs/beatPDivec/v1_autoenc

sOut=/export/c08/lmorove1/kaldi/egs/beatPDivec/trem_noinact_auto30_240fl/exp/
sDirFeats=/export/c08/lmorove1/kaldi/egs/beatPDivec/trem_noinact_auto30_240fl

for ivecDim in 350 400 450 500 550 600 650 700; do
    echo Working on ${ivecDim}
    ./runSVRFold.sh ${sOut} $ivecDim $sDirFeats

    sDirRes=${sOut}/ivec_${ivecDim}/
    sDirOut=${sOut}/ivec_${ivecDim}
    ./evaluate_global_everyone_SVR.sh $sDirRes $sDirOut 
done

echo DONE

#echo Working on trem trem_noinact_auto30_320fl

#sOut=/export/c08/lmorove1/kaldi/egs/beatPDivec/v1_autoenc/exp/
#sDirFeats=/export/c08/lmorove1/kaldi/egs/beatPDivec/v1_autoenc

#sOut=/export/c08/lmorove1/kaldi/egs/beatPDivec/trem_noinact_auto30_320fl/exp/
#sDirFeats=/export/c08/lmorove1/kaldi/egs/beatPDivec/trem_noinact_auto30_320fl

#for ivecDim in 350 400 450 500 550 600 650 700; do
#    echo Working on ${ivecDim}
#    ./runSVRFold.sh ${sOut} $ivecDim $sDirFeats

#    sDirRes=${sOut}/ivec_${ivecDim}/
#    sDirOut=${sOut}/ivec_${ivecDim}
#    ./evaluate_global_everyone_SVR.sh $sDirRes $sDirOut
#done

#echo DONE
#SOut=/export/c08/lmorove1/kaldi/egs/beatPDivec/v1_dysk_auto/exp/
#sDirFeats=/export/c08/lmorove1/kaldi/egs/beatPDivec/v1_dysk_auto

#sOut=/export/c08/lmorove1/kaldi/egs/beatPDivec/v1_3ax_10mfcc_dysk/exp/
#sDirFeats=/export/c08/lmorove1/kaldi/egs/beatPDivec/v1_3ax_10mfcc_dysk

#echo Working on dyskenisia 

#for ivecDim in 350 400 450 500; do
#    echo Working on ${ivecDim}
#    ./runSVRFold.sh ${sOut} $ivecDim $sDirFeats
#done

#echo Working on tremor 

#sOut=/export/c08/lmorove1/kaldi/egs/beatPDivec/v1_trem_auto/exp/
#sDirFeats=/export/c08/lmorove1/kaldi/egs/beatPDivec/v1_trem_auto

#sOut=/export/c08/lmorove1/kaldi/egs/beatPDivec/v1_3ax_10mfcc_tr/exp/
#sDirFeats=/export/c08/lmorove1/kaldi/egs/beatPDivec/v1_3ax_10mfcc_tr

#for ivecDim in 350 400 450 500; do
#    echo Working on ${ivecDim}
#    ./runSVRFold.sh ${sOut} $ivecDim $sDirFeats
#done
