#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate plda2

cmd=run.pl

#export PATH="$HOME/anaconda3/bin:$PATH"
export PATH="./local:$PATH"

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

sDirUtt2spk=$1
sDirMats=$2
sOut=$3

sUtt2spk=${sDirUtt2spk}/utt2spk
cmd=run.pl
$cmd $sOut/featImport.log \
     import_feats.py --utt2spk $sUtt2spk \
     --sDirMats $sDirMats \
     --output-dir $sOut \

conda deactivate
