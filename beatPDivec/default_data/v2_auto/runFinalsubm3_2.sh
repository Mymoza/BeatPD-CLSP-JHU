##########
# This file creates ivector_TrainingFinal ivectors and pkl files on the test subset of the challenge

chmod 755 ./run_Final_auto.sh

stage=0
sDirFeatsTest=/export/b19/mpgill/BeatPD/cis_testing_AE_30ft_orig_inactivity_removed
sDirFeatsTrai=/export/b19/mpgill/BeatPD/AE_30ft_orig_inactivity_removed
ivecDim=650
iNumComponents=50
sKernel=linear
fC=0.2
fEpsilon=0.1

./run_Final_auto.sh $sDirFeatsTrai $sDirFeatsTest $ivecDim $stage $iNumComponents $sKernel $fC $fEpsilon

stage=2
for iNumComponents in 50 100 250 300 400 450 650; do
    for sKernel in 'linear'; do # 'poly' 'sigmoid'; do
	x=-13
	while [ $x -le 2 ]
	do
	    # Convert from scientific to float
	    fC=$(printf "%.14f\n" 2E${x})
	    if [ $iNumComponents -le $ivecDim ]; then
		echo Component is ${iNumComponents}
		fEpsilon=0.1 # default value
		./run_Final_auto.sh $sDirFeatsTrai $sDirFeatsTest $ivecDim $stage $iNumComponents $sKernel $fC $fEpsilon
		
	    fi
	    x=$(( $x + 2))
	done
    done
done
