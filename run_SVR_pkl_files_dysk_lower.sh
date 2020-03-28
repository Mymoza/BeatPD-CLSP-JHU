#!/usr/bin/env bash
chmod 755 /home/sjoshi/codes/python/BeatPD/code/run_auto.sh
chmod 755 /home/sjoshi/codes/python/BeatPD/code/run_SVR_pkl_files_dysk.sh
chmod 755 /home/sjoshi/codes/python/BeatPD/code/runSVRFold.sh

#sOut=/export/c08/lmorove1/kaldi/egs/beatPDivec/v1_dysk_auto/exp/
#sDirFeats=/export/c08/lmorove1/kaldi/egs/beatPDivec/v1_dysk_auto

sOut=/export/c08/lmorove1/kaldi/egs/beatPDivec/v1_3ax_10mfcc_dysk/exp3ax/
sDirFeats=/export/c08/lmorove1/kaldi/egs/beatPDivec/v1_3ax_10mfcc_dysk

echo Working on dyskenisia

for ivecDim in 50 100 150 200 250 300; do
    echo Working on ${ivecDim}
    ./runSVRFold.sh ${sOut} $ivecDim $sDirFeats
done
