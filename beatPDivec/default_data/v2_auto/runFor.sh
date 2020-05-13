#chmod 755 ./run_auto.sh
if [ -f path.sh ]; then . ./path.sh; fi

# Creates and extracts ivectors
ivecDim=350
stage=0
subChallenge=onoff # onoff or tremor or dysk 

sDirFeats=/export/b19/mpgill/BeatPD/AE_60ft_320fl_orig_inactivity_removed

for ((iNumFold=0; iNumFold <=4 ; iNumFold++))  do
    ./run_auto.sh $iNumFold $sDirFeats $ivecDim $stage || exit 1;
done

# compute results for ivecdim 350
sDirRes=`pwd`/exp/ivec_${ivecDim}/
sDirOut=`pwd`/exp/ivec_${ivecDim}

#local/evaluate_global_acc.sh $sDirRes $sDirOut # KNN
local/evaluate_global_SVR.sh $sDirRes $sDirOut # SVR
#local/evaluate_global_everyone_SVR.sh $sDirRes $sDirOut # SVR Everyone 
local/evaluate_global_per_patient_SVR.sh $sDirRes $sDirOut $subChallenge # SVR Per Patient

# Features and UBM are already calculated from the previous code
# So we start at stage 5 as we want to run SVR and Everyone SVR 
stage=2
for  ivecDim in 400 450 500 550 600 650 700;  do

    for ((iNumFold=0; iNumFold <=4 ; iNumFold++))  do
        ./run_auto.sh $iNumFold $sDirFeats $ivecDim $stage || exit 1;
    done

    sDirRes=`pwd`/exp/ivec_${ivecDim}/
    sDirOut=`pwd`/exp/ivec_${ivecDim}
    #local/evaluate_global_acc.sh $sDirRes $sDirOut # KNN
    local/evaluate_global_SVR.sh $sDirRes $sDirOut # SVR
    #local/evaluate_global_everyone_SVR.sh $sDirRes $sDirOut # SVR Everyone
    local/evaluate_global_per_patient_SVR.sh $sDirRes $sDirOut $subChallenge # SVR Per Patient
done
