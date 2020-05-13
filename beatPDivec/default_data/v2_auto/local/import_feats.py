#!/usr/bin/env python3.7
import os
import sys
import numpy as np
import argparse
import re
from scipy.io import loadmat
import kaldi_io
#sDirOut='/export/c08/lmorove1/kaldi/egs/beatPDivec/v1_autoenc/data'
#ark_scp_output='ark:| copy-feats ark:- ark,scp:'+ sDirOut+'/feats.ark,'+ sDirOut+'/feats.scp'
#sDirMats='/export/b03/sbhati/PD/BeatPD/AE_feats'
#sUtt2Speak="/export/c08/lmorove1/kaldi/egs/beatPDivec/v1_3ax_10mfcc_tr/data/Training_Fold4_x/utt2spk"

def import_feats(sUtt2Speak, sDirMats, sDirOut):

    if not  os.path.isdir(sDirOut):
        os.mkdir(sDirOut)
                    
    
    ark_scp_output='ark:| copy-feats ark:- ark,scp:'+ sDirOut+'/feats.ark,'+ sDirOut+'/feats.scp'
    with kaldi_io.open_or_fd(ark_scp_output,'wb') as f:
        for line in open(sUtt2Speak, 'r'):
            utt, speak = line.split()
            vInd=[m.start() for m in re.finditer('_', utt)] # The code is always between second and third '_'
            if len(vInd)==0:
                sCode=utt
            else:
                sCode=utt[vInd[1]+1:vInd[2]]

            sFile=sDirMats+'/'+sCode+'.mat'
            dmat=loadmat(sFile)
            mat=dmat['feat']
            kaldi_io.write_mat(f, mat, utt)
            #print(dNames[utt])



if __name__ == "__main__":

    parser=argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars='@',
        description='Imports external features to Kaldi.')

    parser.add_argument('--utt2spk',dest='sUtt2Speak', required=True)
    parser.add_argument('--sDirMats',dest='sDirMats', required=True)
    parser.add_argument('--output-dir', dest='sDirOut', required=True)
            
    args=parser.parse_args()
            
    import_feats(**vars(args))

                                
