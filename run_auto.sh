#!/usr/bin/env bash
# Copyright 2015-2017   David Snyder
#                2015   Johns Hopkins University (Author: Daniel Garcia-Romero)
#                2015   Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.
#
# See README.txt for more info on data required.
# Results (EERs) are inline in comments below.

#. ./cmd.sh
#. ./path.sh
#set -e


num_components=256 # Larger than this doesn't make much of a difference.
#expDir=`pwd`/exp
#expDir=/export/c08/lmorove1/kaldi/egs/beatPDivec/v1_autoenc/exp

#if [ -f path.sh ]; then . ./path.sh; fi
#. parse_options.sh || exit 1;

foldN=$1
sDirFeats=$2
ivecDim=$3
stage=$4

expDir=${sDirFeats}/exp

trials=data/Testing_Fold${foldN}/trials


if [ $stage -le 0 ] && [ $foldN -le 0 ];then
    utils/utt2spk_to_spk2utt.pl data/all_ubm > \
				data/all_ubm/spk2utt
    utils/fix_data_dir.sh data/all_ubm
    utils/validate_data_dir.sh --no-text --no-feats data/all_ubm
    local/import_feats.sh data/all_ubm $sDirFeats data/all_ubm
    utils/fix_data_dir.sh data/all_ubm

   # sid/compute_vad_decision.sh --vad-config conf/vad.conf --nj 8 --cmd "$train_cmd" \
#		       data/all_ubm exp/make_vad $vaddir
    
fi

if [ $stage -le 1 ]; then
for name in Training_Fold${foldN} Testing_Fold${foldN}; do

    utils/utt2spk_to_spk2utt.pl data/${name} > \
				data/${name}/spk2utt
    utils/fix_data_dir.sh data/${name}
    utils/validate_data_dir.sh --no-text --no-feats data/${name}
    local/import_feats.sh data/${name} $sDirFeats data/${name}
    utils/fix_data_dir.sh data/${name}
    
done
fi

# Train UBM and i-vector extractor.

if [ $stage -le 0 ] && [ $foldN -le 0 ];then
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
sid/train_ivector_extractor_novad.sh --cmd "$train_cmd --mem 25G" --nj 10 \
				     --ivector-dim $ivecDim \
				     --num-iters 5 ${expDir}/full_ubm_$num_components/final.ubm data/Training_Fold${foldN} \
				     ${expDir}/extractor${foldN}_${ivecDim}

# Extract i-vectors.
sid/extract_ivectors_novad.sh --cmd "$train_cmd --mem 6G" --nj 20 \
			      ${expDir}/extractor${foldN}_${ivecDim} data/Training_Fold${foldN} \
			      ${expDir}/ivec_${ivecDim}/ivectors_Training_Fold${foldN}

sid/extract_ivectors_novad.sh --cmd "$train_cmd --mem 6G" --nj 20 \
			      ${expDir}/extractor${foldN}_${ivecDim} data/Testing_Fold${foldN} \
			      ${expDir}/ivec_${ivecDim}/ivectors_Testing_Fold${foldN}


fi
    

# SCORING

#local/plda_scoring.sh data/all_ubm data/Training_Fold${foldN} data/Testing_Fold${foldN} \
#		      exp/ivectors_all_ubm${foldN} exp/ivectors_Training_Fold${foldN} \
#		      exp/ivectors_Testing_Fold${foldN} $trials exp/scores_iVec_Fold${foldN}

sFileTrai=$expDir/ivec_${ivecDim}/ivectors_Training_Fold${foldN}/ivector.scp
sFileTest=$expDir/ivec_${ivecDim}/ivectors_Testing_Fold${foldN}/ivector.scp

if [ $stage -le 3 ]; then
    sOut=$expDir/ivec_${ivecDim}/resxVecFold${foldN}

    # PLDA

    for iNumComponents in 15 20 30 40 50 100 125 150 175; do

	if [ $iNumComponents -le $ivecDim ]; then
	    local/pca_plda_be_bpd.sh $sFileTrai $sFileTest $sOut $iNumComponents
	fi
    done
    
fi

# KNN
if [ $stage -eq 4 ]; then
	sOut=$expDir/ivec_${ivecDim}/resiVecKNN_Fold${foldN}


	for iNeighbors in 1 2 3; do
	    for iNumComponents in 15 30 50 100 125 150 175 200 225 250 275; do

		if [ $iNumComponents -le $ivecDim ]; then
		    local/pca_knn_be_bpd.sh $sFileTrai $sFileTest $sOut $iNumComponents $iNeighbors
		fi
	    done
	done
fi

if [ $stage -eq 5 ]; then  # Just SVR
	sOut=$expDir/ivec_${ivecDim}/resiVecSVR_Fold${foldN}
	for iNumComponents in 50 100 125 150 175 200 225 250 275 300 350 400 450 500 550; do
		for sKernel in 'linear'; do # 'poly' 'sigmoid'; do
			x=-13
			while [ $x -le 2 ]
			do
				# Convert from scientific to float
				fC=$(printf "%.14f\n" 2E${x})
				if [ $iNumComponents -le $ivecDim ]; then
					echo Component is ${iNumComponents}
					fEpsilon=0.1 # default value
					local/pca_svr_bpd.sh $sFileTrai $sFileTest $sOut $iNumComponents $sKernel $fC $fEpsilon
				fi
				x=$(( $x + 2))
			done
		done
	done
fi

if [ $stage -eq 6 ]; then  # Just SVR
	sOut=$expDir/ivec_${ivecDim}/resiVecEveryoneSVR_Fold${foldN}
        for iNumComponents in 350 400 450 500 550 600 650 700; do
                for sKernel in 'linear'; do # 'poly' 'sigmoid'; do
                        x=-13
                        while [ $x -le 0 ]
                        do
                                # Convert from scientific to float
                                fC=$(printf "%.14f\n" 2E${x})
                                if [ $iNumComponents -le $ivecDim ]; then
                                        echo Component is ${iNumComponents}
                                        fEpsilon=0.1 # default value
                                        local/pca_everyone_svr_bpd.sh $sFileTrai $sFileTest $sOut $iNumComponents $sKernel $fC $fEpsilon
                                fi
                                x=$(( $x + 2))
                        done
                done
        done
fi
