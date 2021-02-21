#!/bin/bash

conda activate keras_tf2
cd ml_dl

python train_AE.py --latent_dim 60 \
	-dlP '{"my_data_path":"/home/sjoshi/codes/python/BeatPD/data/BeatPD/cis-pd.training_data.high_pass/", \
	"remove_inactivity":"False"}' \
	--saveAEFeats --saveFeatDir "/export/b19/mpgill/BeatPD/AE_60ft_400fl_high_pass/"

echo "DOOOOOOOOOOONE TIME  TO DO WITH INACTIVITY RMEOVED" 


conda deactivate
