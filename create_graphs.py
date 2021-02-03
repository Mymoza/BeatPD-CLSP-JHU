#!/usr/bin/env python
# coding: utf-8

# Import required libraries

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import HTML, display

# Imports for the high pass signal
from scipy.signal import butter, freqz, lfilter

# KFold
from sklearn.model_selection import KFold

# Import required modules
from sklearn.preprocessing import StandardScaler

import os.path

# To write WAV File
from scipy.io.wavfile import write

# To make derivative work on multiple CPUs
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import sys
from matplotlib import pyplot

from transform_data import * 


def get_plot_title(idx, df_train_label):
    """
    Create a title that identifies the plotted graph with the measurement_id, 
    subject_id, on_off label, dyskinesia and tremor labels.
    
    Keyword arguments:
    - idx: 
    - df_train_label: DataFrame containing the following columns 
            [measurement_id, subject_id, on_off, tremor, dyskenisia]
    
    Returns: A string concatenating all the values mentioned 
    """
    # Following val_* variables are only used to format a cute title for the charts
    val_subject_id = df_train_label.loc[[idx]]["subject_id"].values[0]
    val_on_off = df_train_label.loc[[idx]]["on_off"].values[0]
    val_dyskinesia = df_train_label.loc[[idx]]["dyskinesia"].values[0]
    val_tremor = df_train_label.loc[[idx]]["tremor"].values[0]
    return "{0} = on_off: {1}, dyskinesia: {2}, tremor: {3}".format(
        val_subject_id, val_on_off, val_dyskinesia, val_tremor
    )

# Source: https://stackoverflow.com/questions/28931224/adding-value-labels-on-a-matplotlib-bar-chart
def add_value_labels(ax, spacing=5):
    """Add labels to the end of each bar in a bar chart.

    Arguments:
        ax (matplotlib.axes.Axes): The matplotlib object containing the axes
            of the plot to annotate.
        spacing (int): The distance between the labels and the bars.
    """

    # For each bar: Place a label
    for rect in ax.patches:
        # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        # Number of points between bar and label. Change to your liking.
        space = spacing
        # Vertical alignment for positive values
        va = 'bottom'

        # If value of bar is negative: Place label below bar
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top
            va = 'top'

        # Use Y value as label and format number with one decimal place
        label = "{:.1f}".format(y_value)

        # Create annotation
        ax.annotate(
            label,                      # Use `label` as label
            (x_value, y_value),         # Place label at end of the bar
            xytext=(0, space),          # Vertically shift label by `space`
            textcoords="offset points", # Interpret `xytext` as offset in points
            ha='center',                # Horizontally center label
            va=va)                      # Vertically align label differently for
                                        # positive and negative values.

def compute_symptoms_occurences_dataframe(df_train_label):
    """
    Computes how many times the symptoms are occuring for a single subject_id 
    
    Keyword arguments:
    - df_train_label: DataFrame containing the following columns 
            [measurement_id, subject_id, on_off, tremor, dyskenisia]
    """
    df_train_label = prepro_missing_values(df_train_label=df_train_label)

    # Group data by subject_id
    df_train_label_subject_id = df_train_label.groupby("subject_id")

    df_occurences = []
    symptoms = ["on_off", "dyskinesia", "tremor"]

    for key, value in df_train_label_subject_id:
        for symptom in symptoms:
            # Pour un patient, prendre les 3 dernieres colonnes, et pour 1 symptome, calculer le nb d'occurences
            counter = (
                df_train_label_subject_id.get_group(key)
                .iloc[:, -3:][symptom]
                .value_counts()
            )

            for symptom_value, symptom_occurence in counter.items():
                df_occurences.append(
                    (
                        {
                            "subject_id": key,
                            "symptom": symptom,
                            "symptom_value": symptom_value,
                            "occurence": symptom_occurence,
                        }
                    )
                )

    df_occurences = pd.DataFrame(
        df_occurences, columns=("subject_id", "symptom", "symptom_value", "occurence")
    )

    return df_occurences, df_train_label_subject_id

def plot_symptoms_occurences(df_occurences, df_train_label_subject_id):
    """
    This function plots the occurences of symptoms according to subject_id 

    Keyword Arguments: 
    - df_occurences: contains the df with occurences computed in compute_symptoms_occurences_dataframe
    - df_train_label_subject_id: contains df_train_label grouped by subject_id 
    """

    # There will be one graph plotted for each patient, for each of the 3 symptoms
    nb_subjects_id = (
        df_occurences.subject_id.nunique()
    )  # nb of unique patients in the label file
    print("Nb subject_id : ", nb_subjects_id)
    height = 30 if nb_subjects_id > 10 else 5
    fig, axes = plt.subplots(
        nrows=nb_subjects_id, ncols=3, figsize=(10, height), sharey=True
    )  # 3 cols for the 3 symptoms

    # Quick fix to plot the graphs at the right place. Starts at -1 because in the first for loop
    # it is incremented
    patient = -1
    
    # Plot for all subject_id 3 bar plots for all the symptoms and their occurences
    # Reminder that NaN values (missing values) were replaced with -1 and are shown as such in the plots
    symptoms = ["on_off", "dyskinesia", "tremor"]
    for key, value in df_train_label_subject_id:
        patient = patient + 1  # value used to position the plots (row)
        symptom_no = 0  # value only used to position the plots (col)
        for symptom in symptoms:

            subject_symptom = " ".join(
                [str(key), symptom]
            )  # variable used to create a title for each plot
            df_train_label_subject_id_symptom = df_train_label_subject_id.get_group(key)[symptom].value_counts()
            order = [-1,0,1,2,3,4]
            df_train_label_subject_id_symptom = df_train_label_subject_id_symptom.reindex(order)
            ax = df_train_label_subject_id_symptom.plot(
                kind="bar",
                x=symptom,
                title=subject_symptom,
                ax=axes[symptom_no], # fix it to this ax=axes[patient, symptom_no] if i'm plotting many 
                sharey=True,
            )
#             ax = df_train_label_subject_id.get_group(key)[symptom].value_counts().plot(
#                 kind="bar",
#                 x=symptom,
#                 title=subject_symptom,
#                 ax=axes[symptom_no], # fix it to this ax=axes[patient, symptom_no] if i'm plotting many 
#                 sharey=True,
#             )
            fig.tight_layout()
            plt.tight_layout()
            symptom_no = symptom_no + 1
            add_value_labels(ax)
        plt.show()
    
def plot_axis_on_top(df_train_label, path_train_data, highpass=False):
    """
    Inspired by "plot traces per subject"
    Source: https://machinelearningmastery.com/how-to-load-and-explore-a-standard-human-activity-recognition-problem/
    """
    fig, ax = pyplot.subplots(figsize=(30, 30))
    for idx in df_train_label.index:
        # We only print 10 accelerometers otherwise it's too much 
        if idx == 10:
            break
        pyplot.subplot(10, 1, idx+1)
        df_train_data = pd.read_csv(path_train_data + df_train_label["measurement_id"][idx] + ".csv")
        
        if highpass:
            X_filtered_data, Y_filtered_data, Z_filtered_data = apply_highpass_filter(df_train_data)

            pyplot.plot(df_train_data.iloc[:,-4], X_filtered_data, '-b', label='Z')
            pyplot.plot(df_train_data.iloc[:,-4], Y_filtered_data, '-g', label='X')
            pyplot.plot(df_train_data.iloc[:,-4], Z_filtered_data, '-m', label='Y')
        else: 
            pyplot.plot(df_train_data.iloc[:,-4], df_train_data.iloc[:,-1], '-b', label='Z')
            pyplot.plot(df_train_data.iloc[:,-4], df_train_data.iloc[:,-3], '-g', label='X')
            pyplot.plot(df_train_data.iloc[:,-4], df_train_data.iloc[:,-2], '-m', label='Y')
    pyplot.legend()
    pyplot.tight_layout()
    pyplot.show()
    
def plot_accelerometer(df_train_label, data_type, path_train_data, 
                       path_accelerometer_plots, path_inactivity=None, filename="", mask_path=None):
    """
    Plots the accelerometer data. There will be 3 subplots for each axis (X, Y, Z)
    
    Keyword arguments: 
    - data_type={cis , real} : It depends on which database is used 
    - path_accelerometer_plots: Path where the accelerometer plots are going to be saved 
    - path_inactivity: Path where the dataframe with inactivity removed are 
    - mask_path: If provided, the mask will be applied (for example, inactivity will be removed)
    """
    # Iterating through all the indexes contained in df_train_label
    for idx in df_train_label.index:
        if mask_path is not None:
            df_train_data = apply_mask(path_train_data=path_train_data,
                               measurement_id=df_train_label["measurement_id"][idx],
                               mask_path=mask_path)
        else:
            # Workaround to save graphs for rotation with meaningful title
#             df_train_data = pd.read_csv(path_train_data + df_train_label["measurement_id"][idx] + "_ang_-15.csv")
#             df_train_data = pd.read_csv(path_train_data + df_train_label["measurement_id"][idx] + "_ang_-32.csv")
#             df_train_data = pd.read_csv(path_train_data + df_train_label["measurement_id"][idx] + "_ang_-3.csv")
#             df_train_data = pd.read_csv(path_train_data + df_train_label["measurement_id"][idx] + "_ang_25_30_bound.csv")
            df_train_data = pd.read_csv(path_train_data + df_train_label["measurement_id"][idx] + ".csv")

        # FIXME: BUG ?  why the following goes to 1000xxx sometimes? It should be max 59xxx
        print("measurement_id : ", df_train_label["measurement_id"][idx])
        # Following val_* variables are only used to format a cute title for the charts
        great_title = get_plot_title(idx, df_train_label)

        # The time doesn't have the same name depending on the data_type
        x_axis_data_type = "t" if data_type == "real" else "Timestamp"

        # Normalize the data
        cols_to_norm = ["x", "y", "z"] if data_type == "real" else ["X", "Y", "Z"]
        df_train_data[cols_to_norm] = df_train_data[cols_to_norm].apply(
            lambda x: (x - x.min()) / (x.max() - x.min())
        )

        df_train_data.plot(x=x_axis_data_type, legend=True, subplots=True, title=great_title)

        # Save plotted graph with the measurement_id as name of the file
        plt.savefig(path_accelerometer_plots + df_train_label["measurement_id"][idx] + '_' + filename + ".png")
        plt.savefig(path_accelerometer_plots + df_train_label["measurement_id"][idx] + '_' + filename + ".pdf")
        
        plt.show()
        plt.clf()
        plt.cla()
        plt.close()