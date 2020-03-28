source /home/sbhati/cmd.sh
pids=(1004 1006 1007 1019 1020 1023 1032 1034 1038 1039 1043 1044 1046 1048 1049 1051)

dir="/export/b03/sbhati/PD/BeatPD/"
savedir="/export/b19/mpgill/BeatPD/logs/"

for pid in ${pids[@]}
do
for kfind in `seq 0 4`
do
log_file=${savedir}/${pid}_${kfind}_removeinactivity.log
out_file=${savedir}/${pid}_${kfind}_removeinactivity.txt
$keras_cmd_all -e $log_file -o $out_file /home/mpgill/keras_run.sh ${dir}/train_kfold.py --pid $pid --KFind $kfind -dlP '{"remove_inactivity": "True"}'
done
done

#for pid in ${pids[@]}
#do
#for kfind in `seq 0 4`
#do
#log_file=${savedir}/${pid}_${kfind}_warmLSTM.log
#out_file=${savedir}/${pid}_${kfind}_warmLSTM.txt
#$keras_cmd_all -e $log_file -o $out_file /home/sbhati/keras_run.sh ${dir}/train_kfold.py --pid $pid --KFind $kfind --warmstart_LSTM
#done
#done

#for pid in ${pids[@]}
#do
#for kfind in `seq 0 4`
#do
#log_file=${savedir}/${pid}_${kfind}.log
#out_file=${savedir}/${pid}_${kfind}.txt
#$keras_cmd_all -e $log_file -o $out_file /home/sbhati/keras_run.sh ${dir}/train_kfold.py --pid $pid --KFind $kfind
#done
#done


#for pid in ${pids[@]}
#do
#for kfind in `seq 0 4`
#do
#log_file=${savedir}/${pid}_${kfind}_warmLSTM_addnoise.log
#out_file=${savedir}/${pid}_${kfind}_warmLSTM_addnoise.txt
#$keras_cmd_all -e $log_file -o $out_file /home/sbhati/keras_run.sh ${dir}/train_kfold.py --pid $pid --KFind $kfind -dlP '{"add_noise": "True"}' --warmstart_LSTM
#done
#done

#for pid in ${pids[@]}
#do
#for kfind in `seq 0 4`
#do
#log_file=${savedir}/${pid}_${kfind}_warmLSTM_addrotation.log
#out_file=${savedir}/${pid}_${kfind}_warmLSTM_addrotation.txt
#$keras_cmd_all -e $log_file -o $out_file /home/sbhati/keras_run.sh ${dir}/train_kfold.py --pid $pid --KFind $kfind -dlP '{"add_rotation": "True"}' --warmstart_LSTM
#done
#done

#-dlP '{"add_noise":"True","add_rotation":"True"}'
