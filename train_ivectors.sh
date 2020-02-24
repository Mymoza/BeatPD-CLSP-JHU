#!/bin/bash
#$ -cwd
#$ -j y -o out_train_ivectors
#$ -e /home/mpgill/err_train_ivectors
#$ -m eas
#$ -pe smp 8
#$ -l mem_free=60G,ram_free=60G
#$ -V
#$ -q g.q

source activate bob_py3
python bobivectors.py
