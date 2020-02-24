#!/usr/bin/env python
# coding: utf-8

# # i-vectors with bob package
#
# Documentation:
# - https://groups.google.com/forum/#!msg/bob-devel/ztz2TcTDH_Y/ISjzx6L1BQAJ
# - https://www.idiap.ch/software/bob/docs/bob/bob.learn.em/stable/guide.html#id29
# - https://www.idiap.ch/software/bob/docs/bob/bob.learn.em/stable/guide.html#session-variability-modeling-with-gaussian-mixture-models
# - https://groups.google.com/forum/#!topic/bob-devel/lFda64dmpjY
#
#
# Questions:
# - Which one to use `bob.learn.em.IVectorTrainer` or the one in `bob.bio.gmm.algorithm.IVector`?
#     - it depends. When you want to implement your own application for i-vector training and evaluation, the bob.learn.em classes should work for you. When you are implementing speaker recognition experiments, bob.bio.gmm is the better choice.

# In[30]:


import bob.learn.em
import bob.bio.gmm
import numpy
import pandas as pd
import numpy as np


# ```
# $ conda create --name bob_py3 --override-channels -c https://www.idiap.ch/software/bob/conda -c defaults bob
# $ conda activate bob_py3
# $ conda config --env --add channels https://www.idiap.ch/software/bob/conda/label/archive
# $ conda config --env --add channels defaults
# $ conda config --env --add channels https://www.idiap.ch/software/bob/conda
# $ conda install bob.bio.gmm
# $ conda install nb_conda_kernels
# ```

data_dir='/home/sjoshi/codes/python/BeatPD/data/BeatPD/'

# # TO be deleted when I can import function from other notebook

def define_data_type(data_type):
    # Setup file names

    '''
    data_type={cis , real}

    If data_type is real, data_real_subtype
    data_real_subtype={smartphone_accelerometer , smartwatch_accelerometer , smartwatch_gyroscope}
    '''
    if data_type=='cis':
        path_train_labels=data_dir+data_type+'-pd.data_labels/'+data_type.upper()+'-PD_Training_Data_IDs_Labels.csv'
        path_train_data=data_dir+data_type+'-pd.training_data/'

    if data_type=='real':
        path_train_labels=data_dir+data_type+'-pd.data_labels/'+data_type.upper()+'-PD_Training_Data_IDs_Labels.csv'
        path_train_data=data_dir+data_type+'-pd.training_data/'+data_real_subtype+'/'

    # Display labels
    df_train_label=pd.read_csv(path_train_labels)
    return path_train_data, df_train_label

'''
Filters df_train_label according to a list of measurement_id we are interested in analyzing

Arguments:
- df_train_label: dataframe with labels
- list_measure_id: list of measurement_id

Returns:
- df_train_label: filtered df_train_label containing only the measurements_id we are interested in
'''
def interesting_patients(df_train_label, list_measurement_id):
    filter_measurement_id = df_train_label.measurement_id.isin(list_measurement_id)

    df_train_label = df_train_label[filter_measurement_id]
    return df_train_label

import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.debug("test")


data_type='cis'

path_train_data, df_train_label = define_data_type(data_type=data_type)

# Filter df_train_label according to the measurement_id we are most interested in
#df_train_label = interesting_patients(df_train_label=df_train_label, list_measurement_id=list_measurement_id)

# Array concatenating the training data so we can train the ubm with everything
nd_train_data = np.empty((0,3), float)
list_train_data = []
for idx in df_train_label.index:
    df_train_data=pd.read_csv(path_train_data+df_train_label["measurement_id"][idx]+'.csv')
    x = df_train_data.iloc[:,-3:]
    normed_x = (x - x.mean(axis=0)) / x.std(axis=0)
    list_train_data.append(normed_x.to_numpy())
    nd_train_data = np.append(nd_train_data, normed_x.to_numpy(), axis=0)


# replace the prior gmm with a code that trains a ubm?
# https://groups.google.com/forum/#!searchin/bob-devel/prior%7Csort:date/bob-devel/WLk1t8ixr0A/RyhfiC0ICAAJ

# Creating a fake prior with 256 gaussians of dimension 3
g = 256
#prior_gmm = bob.learn.em.GMMMachine(g, 3)

# Training UBM
gmm = bob.bio.gmm.algorithm.GMM(number_of_gaussians=g)
gmm.train_ubm(nd_train_data)
gmm.project_ubm(nd_train_data)
prior_gmm = gmm.ubm

# SAVE THE UBM

# The input the the TV Training is the statistics of the GMM
gmm_stats_per_class = []
for d in list_train_data:
    for i in d:
        gmm_stats_container = bob.learn.em.GMMStats(g, 3)
        prior_gmm.acc_statistics(i, gmm_stats_container)
        gmm_stats_per_class.append(gmm_stats_container)


### Finally doing the TV training
subspace_dimension_of_t = 2
ivector_trainer = bob.learn.em.IVectorTrainer(update_sigma=True)
# IVectorMachine: Statistical model for the Total Variability training for more
ivector_machine = bob.learn.em.IVectorMachine(
        prior_gmm, subspace_dimension_of_t, 10e-5)

import multiprocessing.pool
pool = multiprocessing.ThreadPool(8)

# train IVector model
bob.learn.em.train(ivector_trainer, ivector_machine,
                        gmm_stats_per_class, 500, pool=pool)

# Printing the session offset w.r.t each Gaussian component
# Returns the Total Variability matrix, T
print(ivector_machine.t)

print(ivector_machine.ubm)

ivector_machine.save(data_dir+'bob.hdf5')

# ivectors projected
#ivector_machine.project(gmm_stats_per_class)
