# BEATPD 

## Set up the environment : 


```
conda create -n BeatPD python=3.5

source activate BeatPD 

conda install --file requirements.txt
```

Make sure that the Jupyter notebook is running on `BeatPD` kernel. 

If the conda environment isn't showing in Jupyter kernels (Kernel > Change Kernel > BeatPD), run: 
```
ipython kernel install --user --name=BeatPD
```
You will then be able to select `BeatPD` as your kernel. 
