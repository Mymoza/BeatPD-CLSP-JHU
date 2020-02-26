# BEATPD 

## Set up the environment : 


```
$ conda create -n BeatPD python=3.5
$ source activate BeatPD 
$ conda install --file requirements.txt
```

Make sure that the Jupyter notebook is running on `BeatPD` kernel. 

If the conda environment isn't showing in Jupyter kernels (Kernel > Change Kernel > BeatPD), run: 
```
$ ipython kernel install --user --name=BeatPD
```
You will then be able to select `BeatPD` as your kernel. 


# I-vector training 

Documentation: 
- https://groups.google.com/forum/#!msg/bob-devel/ztz2TcTDH_Y/ISjzx6L1BQAJ
- https://www.idiap.ch/software/bob/docs/bob/bob.learn.em/stable/guide.html#id29
- https://www.idiap.ch/software/bob/docs/bob/bob.learn.em/stable/guide.html#session-variability-modeling-with-gaussian-mixture-models
- https://groups.google.com/forum/#!topic/bob-devel/lFda64dmpjY



### Steps used to setup the bob_py3 conda environment

```
$ conda create --name bob_py3 --override-channels -c https://www.idiap.ch/software/bob/conda -c defaults bob
$ conda activate bob_py3
$ conda config --env --add channels https://www.idiap.ch/software/bob/conda/label/archive
$ conda config --env --add channels defaults
$ conda config --env --add channels https://www.idiap.ch/software/bob/conda
$ conda install bob.bio.gmm
$ conda install nb_conda_kernels
```


Questions: 
- Which one to use `bob.learn.em.IVectorTrainer` or the one in `bob.bio.gmm.algorithm.IVector`?
    - it depends. When you want to implement your own application for i-vector training and evaluation, the bob.learn.em classes should work for you. When you are implementing speaker recognition experiments, bob.bio.gmm is the better choice.
    
 We chose to use `bob.learn.em.IVectorTrainer` as advised by a maintainer of bob. 
