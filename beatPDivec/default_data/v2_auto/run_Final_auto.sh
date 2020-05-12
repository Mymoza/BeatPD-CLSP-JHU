#!/usr/bin/env bash
# Copyright 2015-2017   David Snyder
#                2015   Johns Hopkins University (Author: Daniel Garcia-Romero)
#                2015   Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.
#
# See README.txt for more info on data required.
# Results (EERs) are inline in comments below.

. ./cmd.sh
. ./path.sh
set -e


num_components=256 # Larger than this doesn't make much of a difference.
expDir=`pwd`/exp

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

sDirFeatsTrai=$1
sDirFeatsTest=$2
ivecDim=$3
stage=$4
iNumComponents=$5
sKernel=$6
fC=$7
fEpsilon=$8

trials=data/Testing_Fold${foldN}/trials


if [ $stage -le 0 ];then
    utils/utt2spk_to_spk2utt.pl data/all_ubm > \
				data/all_ubm/spk2utt
    utils/fix_data_dir.sh data/all_ubm
    utils/validate_data_dir.sh --no-text --no-feats data/all_ubm
    local/import_feats.sh data/all_ubm $sDirFeatsTrai data/all_ubm
    utils/fix_data_dir.sh data/all_ubm

   # sid/compute_vad_decision.sh --vad-config conf/vad.conf --nj 8 --cmd "$train_cmd" \
#		       data/all_ubm exp/make_vad $vaddir
    
fi

if [ $stage -le 1 ]; then
    #Training
    utils/utt2spk_to_spk2utt.pl data/TrainingFinal > \
				data/TrainingFinal/spk2utt
    utils/fix_data_dir.sh data/TrainingFinal
    utils/validate_data_dir.sh --no-text --no-feats data/TrainingFinal
    local/import_feats.sh data/TrainingFinal $sDirFeatsTrai data/TrainingFinal
    
    utils/fix_data_dir.sh data/TrainingFinal
    
    # Testing
    utils/utt2spk_to_spk2utt.pl data/TestingFinal > \
				data/TestingFinal/spk2utt
    utils/fix_data_dir.sh data/TestingFinal
    utils/validate_data_dir.sh --no-text --no-feats data/TestingFinal
    local/import_feats.sh data/TestingFinal $sDirFeatsTest data/TestingFinal

    utils/fix_data_dir.sh data/TestingFinal

    
fi

# Train UBM and i-vector extractor.

if [ $stage -le 0 ];then
sid/train_diag_ubm_novad.sh --cmd "$train_cmd --mem 20G" \
  --nj 20 --num-threads 8 \
  data/all_ubm $num_components \
  ${expDir}/diag_ubm_$num_components

sid/train_full_ubm_novad.sh --nj 40 --remove-low-count-gaussians false \
  --cmd "$train_cmd --mem 25G" data/all_ubm \
  ${expDir}/diag_ubm_$num_components ${expDir}/full_ubm_$num_components
fi

if [ $stage -le 2 ]; then
# Extract i-vectors.
    sid/train_ivector_extractor_novad.sh --cmd "$train_cmd --mem 25G" --nj 4 --num-threads 2 \
					 --ivector-dim $ivecDim \
					 --num-iters 5 ${expDir}/full_ubm_$num_components/final.ubm data/TrainingFinal \
					 ${expDir}/extractorFinal_${ivecDim}

fi

if [ $stage -le 3 ]; then
# Extract i-vectors.
sid/extract_ivectors_novad.sh --cmd "$train_cmd --mem 6G" --nj 5 \
			      ${expDir}/extractorFinal_${ivecDim} data/TrainingFinal \
			      ${expDir}/ivec_${ivecDim}/ivectors_TrainingFinal

sid/extract_ivectors_novad.sh --cmd "$train_cmd --mem 6G" --nj 5 \
			      ${expDir}/extractorFinal_${ivecDim} data/TestingFinal \
			      ${expDir}/ivec_${ivecDim}/ivectors_TestingFinal


fi
    

# SCORING

#local/plda_scoring.sh data/all_ubm data/Training_Fold${foldN} data/Testing_Fold${foldN} \
#		      exp/ivectors_all_ubm${foldN} exp/ivectors_Training_Fold${foldN} \
#		      exp/ivectors_Testing_Fold${foldN} $trials exp/scores_iVec_Fold${foldN}

sFileTrai=${expDir}/ivec_${ivecDim}/ivectors_TrainingFinal/ivector.scp
sFileTest=${expDir}/ivec_${ivecDim}/ivectors_TestingFinal/ivector.scp

#FIXME: I think we can remove foldN everywhere? This variable is not used
if [ $stage -le 5 ]; then  # Just SVR
    sOut=$expDir/ivec_${ivecDim}/resiVecSVR_Fold_all${foldN}
    echo Component is ${iNumComponents}
    local/pca_svr_bpd2.sh $sFileTrai $sFileTest $sOut $iNumComponents $sKernel $fC $fEpsilon

fi
