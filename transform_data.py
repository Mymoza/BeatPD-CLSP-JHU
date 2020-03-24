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

############################################
############ Preprocessing ############
############################################

def prepro_missing_values(df_train_label):
    """
    Filling NaN values with -1. 
    
    Keyword arguments:
    - df_train_label: DataFrame containing the following columns 
            [measurement_id, subject_id, on_off, tremor, dyskenisia]
    """
    # Replace NaN values with -1.0 because otherwise plotting triggers an error
    df_train_label = df_train_label.fillna(value=-1.0)
    return df_train_label


############################################
############ Processing the data ############
############################################

def define_data_type(data_type, data_dir, data_subset='training_data', data_real_subtype=None):
    """
    Setup file names
    
    Keyword arguments:
    - data_type = {cis , real}
    - data_subset: {'training_data', 'ancillary_data', 'testing_data'} to switch between subsets
    - data_real_subtype: only provided for REAL-PD 

    If data_type is real, data_real_subtype will have to be declared as well 
    data_real_subtype={smartphone_accelerometer, smartwatch_accelerometer, smartwatch_gyroscope}
    """
    if data_subset == "testing_data": 
        path_train_labels = (
            data_dir
            + data_type
            + "-pd.data_labels/"
            + data_type.upper()
            + "-PD_"
            + "Test_Data"
            +"_IDs_Labels.csv"
        )  
        path_train_data = data_dir + data_type + "-pd."+data_subset+"/"
    elif data_type == "cis" or data_type == "real":
        # CIS-PD_Training_Data_IDs_Labels.csv
        # CIS-PD_Ancillary_Data_IDs_Labels.csv
        path_train_labels = (
            data_dir
            + data_type
            + "-pd.data_labels/"
            + data_type.upper()
            + "-PD_"
            + ("Training_Data" if data_subset == 'training_data' else "Ancillary_Data")
            +"_IDs_Labels.csv"
        )
        path_train_data = data_dir + data_type + "-pd."+data_subset+"/"
        
    if data_type == "real":
        print('data_dir : ', data_dir)
        print('data_type : ', data_type)
        print('data_subset : ', data_subset)
        print('data_real_subtype :', data_real_subtype)
        path_train_data = data_dir + data_type + "-pd."+data_subset+"/" + data_real_subtype + "/"
        
    # Display labels
    df_train_label = pd.read_csv(path_train_labels)
    return path_train_data, df_train_label

def interesting_patients(df_train_label, list_measurement_id):
    """
    Filters df_train_label according to a list of measurement_id we are interested in analyzing

    Keyword Arguments:
    - df_train_label: Labels DataFrame containing the following columns 
            [measurement_id, subject_id, on_off, tremor, dyskenisia]
    - list_measurement_id: list of measurement_id 

    Returns:
    - df_train_label: filtered df_train_label containing only the measurements_id we are interested in 
    """

    filter_measurement_id = df_train_label.measurement_id.isin(list_measurement_id)

    df_train_label = df_train_label[filter_measurement_id]

    return df_train_label

############################################
############ DERIVATIVE ############
############################################


def get_derivative_value(df_train_data_context, m):
    """
    TODO 
    
    Keyword arguments:
    - df_train_data_context: TODO
    - m: TODO 
    """
    # Dot Product is the sum of the point wise multiplications between a and m
    cij = np.dot(df_train_data_context, m)
    denum = np.dot(m, m)
    return cij / denum

def get_first_derivative(measurement_id, path_train_data, derivative_path, n_zero=3, padding=False, mask_path=None):
    """
    TODO 
    
    cis-pd.training_data.velocity_original_data: Original data, which means the inactivity is untouched 
    
    Keyword arguments:
    - path_train_data: TODO
    - df_train_label: TODO
    - derivative_path: TODO
    - n_zero: TODO
    - Padding: [False, True] 
      - If True, it will add a padding of [0,0,0] at the beginning and at the end of the training data
      - If False, it will just use the existing values of the training data to have a (7,) vector from the
        training data
    - mask_path: Optinal. If provided, it will apply the high pass mask on the training data 
    """
    # m is a vector. For n_zero, it will be [-3, -2, -1, 0, 1, 2, 3]
    m = np.linspace(-n_zero, n_zero, num=2 * n_zero + 1)

    file_path= derivative_path + measurement_id + '.csv'
    if os.path.isfile(file_path):
        print ("File exist : ", file_path)
        return

    # Load the training data
    if mask_path is not None:
        df_train_data = apply_mask(path_train_data, measurement_id, mask_path)
    else: # Going to get the first derivative from the original data 
        try:
            df_train_data = pd.read_csv(path_train_data + measurement_id + ".csv")
        except FileNotFoundError:
            print('Skipping ' + df_train_label["measurement_id"][idx] +
                  ' as it doesn\'t exist for ' +
                  data_real_subtype)

    df_velocity = []

    if padding:
        # Padding DataFrame to add 3 empty rows at the beginning and at the end of the training data
        df_padding = []

        # FIXME: This could probably be made faster but I won't lose time on this
        df_padding.insert(0, {"Timestamp": -1, "X": 0, "Y": 0, "Z": 0})
        df_padding.insert(0, {"Timestamp": -1, "X": 0, "Y": 0, "Z": 0})
        df_padding.insert(0, {"Timestamp": -1, "X": 0, "Y": 0, "Z": 0})

        df_train_data = pd.concat(
            [pd.DataFrame(df_padding), df_train_data], ignore_index=True
        )
        df_padding = pd.DataFrame({"Timestamp": [-1, -1, -1],
                                    "X": [0, 0, 0],
                                    "Y": [0, 0, 0],
                                    "Z": [0, 0, 0]})

        df_train_data_padding = df_train_data.append(df_padding, ignore_index=True)
    else:  # FIXME remove this it's just a quickfix because there is the option with or without padding
        # and the next loop has to be on the original dataframe withtout padding
        df_train_data_padding = df_train_data

    # BUG : This df_velocity contains 3 extra rows. I'm not sure where they come from 
    for row in df_train_data[["X", "Y", "Z"]].itertuples():
        end = row.Index + n_zero

        start = row.Index - n_zero

        # QUICKFIX to the padding and pointers issue 
        if start == -3:
            start = 0
            end = 6
        elif start == -2:
            start = 1
            end = 7
        elif start == -1:
            start = 2
            end = 8
        elif end > len(df_train_data) - 1 and not padding:
            end = len(df_train_data)
            start = len(df_train_data) - (2 * n_zero)

        df_velocity.append([get_derivative_value(df_train_data_padding.loc[start:end, "X"], m),
                             get_derivative_value(df_train_data_padding.loc[start:end, "Y"], m),
                             get_derivative_value(df_train_data_padding.loc[start:end, "Z"], m)])

    # Build the DataFrame with all the columns together so we can save it to CSV
    df_velocity = pd.DataFrame(df_velocity, columns=['X','Y','Z'])

    df_velocity.to_csv(
        derivative_path + measurement_id + ".csv",
        index=False,
        header=["X_velocity", "Y_velocity", "Z_velocity"],
    )
    
############################################
############ HIGH PASS FILTER ############
############################################

# Source of the code about the highpass: https://gist.github.com/junzis/e06eca03747fc194e322

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def apply_highpass_filter(df_train_data):
    # Filter requirements.
    order = 10
    fs = 50.0  # sample rate, Hz
    cutoff = 0.5  # 3.667  # desired cutoff frequency of the filter, Hz
    
    # Get the filter coefficients so we can check its frequency response.
    b, a = butter_highpass(cutoff, fs, order)
    
    # Filter the data
    # X = [:,-3], Y = [:,-2], Z = [:,-1]
    X_filtered_data = butter_highpass_filter(df_train_data.iloc[:,-3], cutoff, fs, order)
    Y_filtered_data = butter_highpass_filter(df_train_data.iloc[:,-2], cutoff, fs, order)
    Z_filtered_data = butter_highpass_filter(df_train_data.iloc[:,-1], cutoff, fs, order)
    
    return X_filtered_data, Y_filtered_data, Z_filtered_data

def high_pass_filter(df_train_label, high_pass_path, path_train_data, data_type):
    """
    Apply a high pass filter to the measurement_id files in df_train_label and write the result 
    in a folder. 
    
    Keyword arguments: 
    - df_train_label: DataFrame containing the following columns 
            [measurement_id, subject_id, on_off, tremor, dyskenisia]
    - high_pass_path: Path to the file where the files will be written 
    - path_train_data: TODO
    - data_type: TODO 
    """
    df_high_pass = []

    # Load every training file for each "row of labels" we have loaded in df_train_label
    for idx in df_train_label.index:
        # Load the training data
        try:
            df_train_data = pd.read_csv(path_train_data + df_train_label["measurement_id"][idx] + ".csv")
            print('Working on ', df_train_label["measurement_id"][idx])
        except FileNotFoundError:
            print('Skipping ' + df_train_label["measurement_id"][idx] +
                  ' as it doesn\'t exist for the subtype')
            continue

        # Filter the data
        # X = [:,-3], Y = [:,-2], Z = [:,-1]
        # Transformed to DataFrame for the purpose of writing to csv and transposed to have it as a column 
        X_filtered_data, Y_filtered_data, Z_filtered_data = apply_highpass_filter(df_train_data)
        
        # Merge the dataframes together 
        df_high_pass =  pd.DataFrame(np.vstack([X_filtered_data,
                                                Y_filtered_data,
                                                Z_filtered_data]).T,columns= ["X", "Y", "Z"])
        
        # Save to a folder 
        df_high_pass.to_csv(
            high_pass_path + df_train_label["measurement_id"][idx] + ".csv",
            index=False,
            #header=False,
        )
        

# TODO : Refactor so it calls the function that only do highpass 
# TODO: Refactor so that it's not doing plots 
def remove_inactivity_highpass(
    df_train_label,
    path_train_data,
    data_type,
    energy_threshold,
    duration_threshold,
    mask_path,
    plot_frequency_response=False,
    plot_accelerometer_after_removal=False,
):
    """
    Removes inactivity according to a high pass filter. It will only be applied to the measurement_id provided
    in the df_train_label variable. A first condition for a measurement to be removed at a certain timestamp
    is that first, the energy is less than the energy_treshold. After that, we identify candidates with a vector
    where 0 represents a timestamp we want to keep, and 1 represents timestamps we detected as below the minimum 
    energy threshold. 
    
    The second threshold, called duration_threshold, represents the condition that there must be a minimum 
    number of consecutives candidates to be removed befoer the candidates will be indeed removed and confirmed
    as inactivity. For example, we could decide to only remove sections that are at least 1 minute long of 
    inactivity detected.
    
    Keyword Arguments:
    - df_train_label: DataFrame containing the following columns 
            [measurement_id, subject_id, on_off, tremor, dyskenisia]
    - path_train_data: TODO
    - data_type: TODO 
    - energy_threshold: what percentage of the max energy do we consider as inactivity?
        For example, 1 of the max is considered as inactivity
    - duration_threshold: how long do we want to have inactivity before we remove it? 
        For example 3000x0.02ms=1min of inactivity minimum before those candidates are removed
    - mask_path: Path where to save the mask 
    - plot_frequency_response: Optional. {True, False}. 
                               Flag to determine if we want to plot the frequency response or not
    - plot_accelerometer_after_removal: Optinal. {True, False}.
                                Flag to determine if we want to plot the accelerometer after the inactivity
                                is removed
    """
    # Filter requirements.
    order = 10
    fs = 50.0  # sample rate, Hz
    cutoff = 0.5  # 3.667  # desired cutoff frequency of the filter, Hz

    # Get the filter coefficients so we can check its frequency response.
    b, a = butter_highpass(cutoff, fs, order)

    # Load every training file for each "row of labels" we have loaded in df_train_label
    for idx in df_train_label.index:
        # Load the training data
        try:
            df_train_data = pd.read_csv(path_train_data + df_train_label["measurement_id"][idx] + ".csv")
            print('Working on ', df_train_label["measurement_id"][idx])
        except FileNotFoundError:
            print('Skipping ' + df_train_label["measurement_id"][idx] +
                  ' as it doesn\'t exist for the subtype')
            continue
        # Set the time axis. It's not the same name for the two databases
        x_axis_data_type = "t" if data_type == "real" else "Timestamp"
        t = df_train_data[x_axis_data_type]

        # Filter the data
        # X = [:,-3], Y = [:,-2], Z = [:,-1]
        X_filtered_data, Y_filtered_data, Z_filtered_data = apply_highpass_filter(df_train_data)

        ### Following section works on removing inactivity following a treshold
        # Get the absolute max values for X, Y, Z
        # FIXME: This could be made better but I won't lose time on this now
        # Get in a Series format because that's what the threshold function is expecting
        max_values = pd.Series(np.array([
                    np.abs(X_filtered_data).max(),
                    np.abs(Y_filtered_data).max(),
                    np.abs(Z_filtered_data).max(),
                ]))

        # Get the threshold of the filtered data
        df_treshold = get_df_threshold(energy_threshold, max_values)
#         print('df threshold : ', df_treshold)
        # Get 0/1 candidates
        X_zeros = np.abs(X_filtered_data) <= df_treshold[0]
        Y_zeros = np.abs(Y_filtered_data) <= df_treshold[1]
        Z_zeros = np.abs(Z_filtered_data) <= df_treshold[2]
        
#         display(X_zeros)
#         display(Y_zeros)
#         display(Z_zeros)
        
        # Change data from boolean to 0 and 1s (int)
        X_zeros = X_zeros.astype(int)
        Y_zeros = Y_zeros.astype(int)
        Z_zeros = Z_zeros.astype(int)
        # AND operand to identify candidates across all axis
        # FIXME: change name of variable bc it's not a df
        df_zeros = X_zeros & Y_zeros & Z_zeros


        # Check if it reaches the time treshold (like minimum 1 minute long to be removed)
        start = 0  # Start and end of the series of 1
        end = 0
        indices_list = []  # List of tuples
        howmany = 0  # How many groups we identified (not required, just nice metric)
        count = 0  # How many 1 in a row we found

        # Counts the number of 0s and 1s in the original data before we apply the threshold
#         unique, counts = np.unique(df_zeros, return_counts=True)
#         print(dict(zip(unique, counts)))

        # Change the candidates for removal (1) to 0 if there are not enough 1s in a row to reach the
        # threshold. For example there needs to be 3000 times 1s in a row for 1 minute of "inactivity"
        # to be removed
        for i in range(0, len(df_zeros)):
            if df_zeros[i] == 1:
                count = count + 1
            else:
                if count >= duration_threshold:
                    start = i - count
                    end = i - 1
                    indices_list.append((start, end))
                    howmany = howmany + 1
                    count = 0
                elif (
                    count >= 1
                ):  # if it doesn't reach the threshold, we change the 1 for 0 because we don't want to remove those
                    start = i - count
                    end = i
                    df_zeros[start:end] = [0] * (end - start)
                    count = 0

        #print("There are "+ str(howmany) + " groups identified as candidates to be removed")

        # Counts the number of 0s and 1s in the data after we applied the threshold
#         unique, counts = np.unique(df_zeros, return_counts=True)
#         print(dict(zip(unique, counts)))

        # Save 0/1 candidates to csv
        # I use 1-df_zeros to swap the 0s and 1s.
        # 1: we want to keep this measure
        # 0: detected as inactivity so we want to remove it
        # Previously, it was the opposite, as 1s were considered inactivity
        df_mask_highpass = pd.DataFrame(1 - df_zeros)
        df_mask_highpass.to_csv(
            mask_path + df_train_label["measurement_id"][idx] + ".csv",
            index=False,
            header=False,
        )
        
        

        # Plot the accelerometer with the removed sections
        # The [0] is used to get a pandas.Series instead of a DataFrame
        # We insert Timestamp again as it was removed for the filtering
        # BUG : Should I use df_train_data or the filtered high pass dataframe?
        if plot_accelerometer_after_removal:
            filtered_df = df_train_data.iloc[:, -3:].multiply(df_mask_highpass[0], axis=0)
            print('LEN FILTERED DF : ', len(filtered_df))
            filtered_df.insert(0, x_axis_data_type, df_train_data[x_axis_data_type])
            great_title = get_plot_title(idx, df_train_label)
            filtered_df.plot(
                x=x_axis_data_type, legend=True, subplots=True, title=great_title
            )
            display(filtered_df)
            plt.show()
            plt.clf()
            plt.cla()
            plt.close()

        # Plot the frequency response, and plot both the original and filtered signals for X, Y and Z.
        if plot_frequency_response:
            # TODO: Make the graphs bigger
            w, h = freqz(b, a, worN=8000)
            plt.subplot(4, 1, 1)
            plt.plot(0.5 * fs * w / np.pi, np.abs(h), "b")
            plt.plot(cutoff, 0.5 * np.sqrt(2), "ko")
            plt.axvline(cutoff, color="k")
            plt.xlim(0, 0.5 * fs)
            plt.title("Highpass Filter Frequency Response")
            plt.xlabel("Frequency [Hz]")
            plt.grid()

            # Plot X
            plt.subplot(4, 1, 2)
            plt.plot(t, df_train_data.iloc[:,-3], "b-", label="X")
            plt.plot(t, X_filtered_data, "g-", linewidth=2, label="filtered X")
            plt.legend()
            plt.grid()

            # Plot Y
            plt.subplot(4, 1, 3)
            plt.plot(t, df_train_data.iloc[:,-2], "b-", label="Y")
            plt.plot(t, Y_filtered_data, "g-", linewidth=2, label="filtered Y")
            plt.legend()
            plt.grid()

            plt.subplot(4, 1, 4)
            plt.plot(t, df_train_data.iloc[:,-1], "b-", label="Z")
            plt.plot(t, Z_filtered_data, "g-", linewidth=2, label="filtered Z")
            plt.legend()

            plt.xlabel("Time [sec]")
            plt.grid()

            plt.subplots_adjust(hspace=0.7)
            plt.show()

            plt.show()
            plt.clf()
            plt.cla()
            plt.close()

############################################
############ INACTIVITY REMOVAL ############
############################################

def get_df_threshold(threshold_energy, max_values):
    """
    This function returns a dataframe of shape (3,1) containing what is the treshold for each X,Y,Z axis 
    depending on threshold_energy. 

    For example if threshold_energy=10, the function is going to return what is 10% of the max_values

    Arguments:
    - threshold_energy: Percentage of the max values we want to use as treshold 
    - max_values: Dataframe of the max values 
    """
    return (max_values * threshold_energy) / 100

def apply_mask(path_train_data, measurement_id, mask_path):
    """
    Apply a mask on the list of measurement_ids provided through df_train_label 
    
    Keyword arguments:
    - TODO
    """
    # Load the training data
    try:
        df_train_data = pd.read_csv(path_train_data + measurement_id + ".csv")
        # smartwatch_accelerometer and smartwatch_gyroscope have an additional device_id column
        # in the training data and we want to remove it 
        df_train_data = df_train_data.drop(['device_id'], axis=1, errors='ignore')
        print('PATH !!! : ', mask_path + measurement_id + ".csv")
        df_mask = pd.read_csv(mask_path + measurement_id + ".csv", header=None)
        # multiply df_train_data by mask so the values to be removed are at 0
        df_train_data.iloc[:, -3:] = np.multiply(df_train_data.iloc[:, -3:], df_mask)#[:, -1:])

        #         display(df_train_data)

        # Drop the 0 values from the training DataFrame
        df_train_data = df_train_data[(df_train_data.iloc[:, -3:].T != 0).any()]
        df_train_data.reset_index(drop=True, inplace=True)
    except FileNotFoundError:
        print('Skipping ' + measurement_id +
              ' as it doesn\'t exist the subtype')
        pass
        
    return df_train_data


#### Remove inactivity and mean offset to make the data at 0 

def remove_inactivity_mean_offset(df_train_label):
    """
    Removal of inactivity segments detected with a threshold after the energy is centered over the mean
    This function is expected to be less efficient than removal of inactivity with the highpass threshold 
    
    Keyword arguments: 
    - df_train_label: DataFrame containing the following columns 
            [measurement_id, subject_id, on_off, tremor, dyskenisia]
    """
    for idx in df_train_label.index:
        df_train_data = pd.read_csv(
            path_train_data + df_train_label["measurement_id"][idx] + ".csv"
        )

        ### Following section works on removing offset with mean
        inputData_DCRemoved = df_train_data.iloc[:, -3:] - df_train_data.iloc[
            :, -3:
        ].mean(axis=0)

        # inputData with DC Removed only has NaN for the Timestamp values, so we are adding them
        inputData_DCRemoved.insert(0, "Timestamp", df_train_data["Timestamp"])

        # Plot the graph
        great_title = get_plot_title(idx, df_train_label)
        print(df_train_label["measurement_id"][idx])
        inputData_DCRemoved.plot(
            x="Timestamp", legend=True, subplots=True, title=great_title
        )

        ### Following section works on removing inactivity following a treshold
        # Get the absolute max values for X, Y, Z
        max_values = inputData_DCRemoved.iloc[:, -3:].abs().max()

        # Compute what is X% of that max
        df_treshold = get_df_threshold(10, max_values)

        df_candidates = inputData_DCRemoved[
            (inputData_DCRemoved.X.abs() <= df_treshold["X"])
            & (inputData_DCRemoved.Y.abs() <= df_treshold["Y"])
            & (inputData_DCRemoved.Z.abs() <= df_treshold["Z"])
        ]

        print("Candidates to be removed:")
        display(df_candidates)

        filter_df = inputData_DCRemoved[
            ~inputData_DCRemoved.isin(df_candidates)
        ].dropna(how="all")
        great_title = "filter_df for : " + great_title
        filter_df.plot(x="Timestamp", legend=True, subplots=True, title=great_title)

        # FIXME: This function is not done. As it's not the priority, i'm switching to work on highpass filter
        # It doesn't remove the identified candidates or save the 0/1 vector

# This didn't work because it's using pct_change between X coordinates where coincidences happens


def remove_inactivity_pct_change(df_train_label, data_dir, path_train_data, data_type, data_real_subtype=""):
    """
    Save .csv files with silence (inactivity) removed 

    Path used: 
    # cis-pd.training_data.no_silence/
    # real-pd.training_data.no_silence/smartphone_accelerometer/
    # real-pd.training_data.no_silence/smartwatch_accelerometer/
    # real-pd.training_data.no_silence/smartwatch_gyroscope/
    # data_type = {'cis', 'real'}

    Arguments:
    df_train_label: Dataframe with training labels

    data_real_subtype: Optional. If data_type is real, data_real_subtype needs to be provided
        data_real_subtype={smartphone_accelerometer , smartwatch_accelerometer , smartwatch_gyroscope}

    Returns: 
    path_no_inactivity_data: Return the path where the files are saved because it is needed
                          if we want to plot the accelerometer, for example
    """
    count = 0
    for idx in df_train_label.index:
        df_train_data = pd.read_csv(
            path_train_data + df_train_label["measurement_id"][idx] + ".csv"
        )
        # print('measurement id : ', df_train_label["measurement_id"][idx])
        # display(df_train_data)
        cols_to_norm = ["x", "y", "z"] if data_type == "real" else ["X", "Y", "Z"]
        df_train_data[cols_to_norm] = df_train_data[cols_to_norm].apply(
            lambda x: (x - x.min()) / (x.max() - x.min())
        )
        periods = 300
        df_pct_change = df_train_data.iloc[:, -3:].pct_change(periods=periods)
        display(df_pct_change)
        df_pct_change.columns = ["X", "Y", "Z"]
        # print('pct_change measurement id : ', df_train_label["measurement_id"][idx])
        # display(df_pct_change)

        # Apply the treshold to the DataFrame with an AND condition, so all axis must have at least 1% of change
        # between the periods
        # pd.options.display.max_rows = 1000
#         print("----------before filter--------")
#         display(df_pct_change.abs())

#         print("WHAT IS DETECTED AS INACTIVITY")
#         display(
#             df_pct_change[
#                 (df_pct_change.X.abs() < 0.002)
#                 | (df_pct_change.Y.abs() < 0.002)
#                 | (df_pct_change.Z.abs() < 0.002)
#             ]
#         )
#         print("END OF WHAT IS DETECTED AS INACTIVITY")

        df_pct_change = df_pct_change[
            (df_pct_change.X.abs() >= 0.002)
            & (df_pct_change.Y.abs() >= 0.002)
            & (df_pct_change.Z.abs() >= 0.002)
        ]
#         print("----------after filter--------")
#         display(df_pct_change)

        filter_df = df_train_data[
            df_train_data.index.isin(df_pct_change.index)#.to_list())
        ]

        # Counts the number of time where we had to remove inactivity from a dataframe to know how often
        # the inactivity zones appear.
        print("len(filter_df)+periods ", str(len(filter_df) + periods))
        print("len(df_train_data) ", str(len(df_train_data)))
        if len(filter_df) + periods != len(df_train_data):
            count = count + 1

        # To provide the name of the header for the Dataframe, we get the name of the x axis as it depends
        # on the data_type and then we insert it at the first position before the X,Y,Z axis
        x_axis_data_type = "t" if data_type == "real" else "Timestamp"
        cols_to_norm.insert(0, x_axis_data_type)

#         filter_df.plot(x='Timestamp',legend=True, subplots=True,title='allo')

        # Save the dataframe in a file with the measurement_id as the name of the file
        path_no_inactivity_data = (
            data_dir
            + data_type
            + "-pd.training_data.no_silence/"
            + data_real_subtype
            + "/"
        )
        filter_df.to_csv(
            path_no_inactivity_data + df_train_label["measurement_id"][idx] + ".csv",
            index=False,
            header=cols_to_norm,
        )
    print(
        "Inactivity zones were detected ",
        str(count),
        " times out of ",
        str(len(df_train_label.index)),
    )
    return path_no_inactivity_data

############################################
############ WRITE WAV FILE ############
############################################

def write_wav(measurement_id, path_train_data, wav_path, mask_path, sAxis):
    """
    Write a wav file 
    
    Keyword arguments:
    - measurement_id: Measurement id of the file we want to write to wav format 
    - wav_path: Path where to save the wav file 
    - mask_path: Path where to apply the mask to the wav file 
    - axis: {'X','Y','Z'}. String. Uppercase. One of the axis
    """
    
    file_path= wav_path + measurement_id + '.wav'
    print('File path : ', file_path)
    if os.path.isfile(file_path):
        # FIX ME: it doesn't go here?
        print ("File exist : ", file_path)
    else:
        df_train_data = apply_mask(path_train_data,
                                   measurement_id,
                                   mask_path)

        # Save to WAV
        samplerate = 8000 # Hz, we need 8kHz for Kaldi to be happy 

        # We're only writing to WAV file for one of the axis 
        if sAxis == 'X':
            df_train_data_axis = df_train_data.iloc[:,1:2]
        elif sAxis == 'Y':
            df_train_data_axis = df_train_data.iloc[:,2:3]
        elif sAxis == 'Z': 
            df_train_data_axis = df_train_data.iloc[:,3:4]
        write(wav_path +
              measurement_id + '.wav', samplerate, df_train_data_axis.to_numpy())


def create_cis_wav_files(data_subset, data_dir, sAxis, data_type="cis"):
    """
    Create wav files for the CIS-PD database 
    
    Keyword arguments:
    - data_subset: {'training_data', 'ancillary_data', 'testing_data'}
    - data_dir:
    - sAxis: {'X', 'Y', 'Z'}
    - data_type="cis" 
    """
    path_train_data, df_train_label = define_data_type(data_type,
                                                       data_dir,
                                                       data_subset)
    do_work = partial(
        write_wav, 
        path_train_data=path_train_data,
        wav_path=data_dir+'cis-pd.'+data_subset+'.wav_'+sAxis+'/',
        sAxis=sAxis,
        mask_path=data_dir+'cis-pd.'+data_subset+'.high_pass_mask/'
    )

    num_jobs = 8
    with ProcessPoolExecutor(num_jobs) as ex:
        results = list(ex.map(do_work, df_train_label['measurement_id']))
    
def create_real_wav_files(data_subset, data_dir, sAxis, data_type="real"): 
    """
    Create wav files for the REAL-PD database 
    
    Keyword arguments:
    - data_subset: {'training_data', 'ancillary_data', 'testing_data'}
    - data_dir:
    - sAxis: {'X', 'Y', 'Z'}
    - data_type="real" 
    """
    for data_real_subtype in ['smartphone_accelerometer', 'smartwatch_accelerometer', 'smartwatch_gyroscope']:
        path_train_data, df_train_label = define_data_type(data_type, data_dir, data_subset, data_real_subtype)
    #     list_mesurement_id=['33f5a031-43a8-496a-89ee-0b9d99019617']
        # Filter df_train_label according to the measurement_id we are most interested in
    #     df_train_label = interesting_patients(df_train_label=df_train_label, list_measurement_id=list_measurement_id)

        for idx in df_train_label.index:
            try:            
                df_train_data = pd.read_csv(path_train_data + df_train_label["measurement_id"][idx] + ".csv")
            except FileNotFoundError:
                print('Removing ' + df_train_label["measurement_id"][idx] +
                      ' as it doesn\'t exist for ' +
                      data_real_subtype)
                df_train_label = df_train_label.drop(idx)

        do_work = partial(
            write_wav, 
            path_train_data=path_train_data,
            wav_path=data_dir+'real-pd.'+data_subset+'.wav_'+sAxis+'/'+data_real_subtype+'/',
            sAxis=sAxis,
            mask_path=data_dir+'/real-pd.'+data_subset+'.high_pass_mask/'+data_real_subtype+'/'
        )

        num_jobs = 8
        with ProcessPoolExecutor(num_jobs) as ex:
            results = list(ex.map(do_work, df_train_label['measurement_id']))