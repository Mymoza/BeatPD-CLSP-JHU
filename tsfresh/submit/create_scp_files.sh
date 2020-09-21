#!/bin/bash

# $1 = name of the scp file to be created
# combhpfnoinact.rotate_3

# $2 = same thing as $1 but for testing 
# combhpfnoinact.rotate_3 

cd data/
cp cis-pd.training.scp cis-pd.training.$1.scp 
cp cis-pd.testing.scp cis-pd.testing.$1.scp

sed -i "s/cis-pd.training_data/cis-pd.training_data.$1/g" cis-pd.training.$1.scp
sed -i "s/cis-pd.testing_data/cis-pd.testing_data.$1/g" cis-pd.testing.$1.scp
