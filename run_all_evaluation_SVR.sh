#!/bin/bash


#./evaluate_global_SVR.sh /export/c08/lmorove1/kaldi/egs/beatPDivec/v1_autoenc/exp/ivec_50/ /export/c08/lmorove1/kaldi/egs/beatPDivec/v1_autoenc/exp/ivec_50/

for ivecDim in 50 100 150 200 250 300; do
    ./evaluate_global_SVR.sh /export/c08/lmorove1/kaldi/egs/beatPDivec/v1_autoenc/exp/ivec_${ivecDim}/ /export/c08/lmorove1/kaldi/egs/beatPDivec/v1_autoenc/exp/ivec_${ivecDim}/
done

for ivecDim in 50 100 150 200 250 300; do
    ./evaluate_global_SVR.sh /export/c08/lmorove1/kaldi/egs/beatPDivec/v1_dysk_auto/exp/ivec_${ivecDim}/ /export/c08/lmorove1/kaldi/egs/beatPDivec/v1_dysk_auto/exp/ivec_${ivecDim}/
done

for ivecDim in 50 100 150 200 250 300; do
    ./evaluate_global_SVR.sh /export/c08/lmorove1/kaldi/egs/beatPDivec/v1_trem_auto/exp/ivec_${ivecDim}/ /export/c08/lmorove1/kaldi/egs/beatPDivec/v1_trem_auto/exp/ivec_${ivecDim}/
done
