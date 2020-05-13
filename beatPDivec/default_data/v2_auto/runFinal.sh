chmod 755 ./run_Final_auto.sh
#on off
stage=5
sDirFeatsTest=/export/b19/mpgill/BeatPD/cis_testing_AE_30ft_orig_inactivity_removed
sDirFeatsTrai=/export/b19/mpgill/BeatPD/AE_30ft_orig_inactivity_removed
ivecDim=450
iNumComponents=400
sKernel=linear
fC=0.2
fEpsilon=0.1

./run_Final_auto.sh $sDirFeatsTrai $sDirFeatsTest $ivecDim $stage $iNumComponents $sKernel $fC $fEpsilon
