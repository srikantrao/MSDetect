"""
Count the number of saccades in the Pupil trace.
Perform logistic regression using the saccade frequency to classify MS vs Non-MS
"""

import numpy as np
import pandas as pd
import streamlit as st
import glob 
import os 
import matplotlib.pyplot as plt 
import re
import scipy.signal
from sklearn.linear_model import LogisticRegressionCV 
from sklearn.metrics import accuracy_score
from utils import pre_processing
from utils import plot_utils
import argparse
from utils import validation

def count_saccades(file_path, num_samples = 4600, 
                   amplitude = 0.0, verbose = False,
                   plot_trace = False):
    """
    Count the Number of X and Y micro-saccades.
    @file_path: Path to the numpy file.
    @num_samples: Number of samples to downsample to. 
    """
    # Get the Head and the Tail of the Path
    head, tail = os.path.split(file_path)
    # name = tail.split("_")
    
    # Load the trace 
    trace = np.load(file_path)
    # Get rid of the first 100 and the last 100 samples 
    trace = trace[:, 100: 4700]
    
    # Smooth the trace 
    # trace = scipy.signal.savgol_filter(trace, 31, 2, axis = 1)

    # Downsample if num_samples is not None 
    if num_samples < 4600:
        trace = pre_processing.downsample_traces(trace, num_samples, axis = 1)
        if verbose:
            print(f"Trace {tail} was down sampled to {trace.shape[1]}")
    
    # Convert Numpy Array to a DataFrame
    df = pd.DataFrame.from_records(trace)
    df = df.transpose()
    df.columns = ['X', 'Y', 'Velocity']
    # Create a Dataframe to keep track of the differences 
    derv = df.diff()
    
    # TODO: Figure out a way to do this without iterating
    fp = False
    fn = False
    values = 0
    indices = []
    for index, row in derv.iterrows():
        # Finite State Machine with 3 states 
        if not fp and not fn and row['X'] >= amplitude:
            fp = True
            values += 1
            indices.append(index)
        elif fp and not fn and row['X'] < 0:
            fp = False
        # Add this state for completeness 
        elif fp and not fn and row['X'] >= amplitude:
            continue 
        elif not fp and not fn and -row['X'] >= amplitude:
            fn = True
            values += 1
            indices.append(index)
        elif not fp and fn and -row['X'] < 0:
            fn = False
        elif not fp and fn and -row['X'] >= amplitude:
            continue

    # values = derv[abs(derv) > amplitude]
    
    if verbose:
        print(f"Number of Saccades: {values}. List of time stamps : {indices}")
    if plot_trace:
        plot_utils.plot_line(df['X'], title = file_path)
        st.pyplot()

    return values


def count_saccades_in_dir(file_dir, amplitude = 0.0,
                          num_samples = 4600, verbose = False,
                          plot_trace = False):
    """
    Count the number of X Saccades in every file
    @param file_dir: File Directory where to look for .npy files 
    @param amplitude: Minimum amplitude to check for micro-saccades
    @param num_samples: Number of samples to use 
    @param verbose: Print additional information helpful for debug.
    @returns: List of tuples -> (file names, number of saccades)
    """
    files = glob.glob(os.path.join(file_dir, "*.npy"))
    num_saccades_list = []
    for fi in files:
        num_saccades_list.append((fi, count_saccades(fi, amplitude = amplitude, num_samples = num_samples, verbose = verbose, plot_trace = plot_trace)))
    
    return num_saccades_list

def build_csv(num_saccades_list, csv_file, append=False, verbose = False):
    """
    Build the csv file which can be used for comparison 
    with hand labelled data.
    @param num_saccades_list - Tuple --> (filename, number of saccades)
    @param csv_file: csv file to which to write everything 
    @param append: Append to the file if a previous csv has been created
    @return: None. Build a csv file and writ
    """
    if append:
        write_mode = 'a'
    else:
        write_mode = "w"

    with open(csv_file, write_mode) as f:
        f.write("subject_id,eye,version,num_saccades\n") 

        for file_name, num_saccades in num_saccades_list:
            file_name = os.path.basename(file_name)
            
            sub_id, ver, eye = file_name.split("_")[0:3]

            line = [sub_id] + ["1" if eye == "L" else "2"] + [ver]
            line.append(str(num_saccades))
            f.write(",".join(line) + "\n")


def build_saccades_csv(file_dir, csv_file, amplitude = 0.0, num_samples = 4600, append = False, verbose = False, plot_trace = False):
    """
    Build a csv which holds information about the number of saccades in each     trace.
    @param file_dir: the directory which contains all the traces
    @param csv_file: Path of the new csv file that needs to be created
    @param amplitude: Minimum amplitude to use to detect micro-saccades
    @param num_samples: Number of samples to use. Input gets downsampled if 
                        below 4600
    @param append: Append to the csv file if set to True
    @param verbose: Print additional information useful for debug 
    """
    
    num_saccades_list = count_saccades_in_dir(file_dir, amplitude = amplitude, num_samples = num_samples, verbose = verbose, plot_trace = plot_trace)

    build_csv(num_saccades_list, csv_file, append = append, verbose = verbose)
    
    if verbose:
        print(f"Micro Saccade frequency has been written to {csv_file}")


def run_logistic_regression(csv_file = "./data/automated_saccades_frequency.csv",
                            patients_file = "/home/shrikant/EnVision/data/updated_patients_stats.csv",                            
                            test_fraction = 0.1,
                            verbose = True):

    # Load the csv file 
    sac_pd = pd.read_csv(csv_file)
    if verbose:
        print("Head of the Saccade Frequency Dataframe")
        print(sac_pd.head())
        print(f"Num Rows: {sac_pd.shape[0]}. Num Cols : {sac_pd.shape[1]}")

    # Load the patients file 
    pat_pd = pd.read_csv(patients_file)
    if verbose:
        print("Head of the Patient Information Dataframe")
        print(pat_pd.head())
        print(f"Num rows: {pat_pd.shape[0]}. Num Cols: {pat_pd.shape[1]}")
    
    # Generate X, y, subject_id 
    X = sac_pd['num_saccades'].values
    subject_id = sac_pd['subject_id'].values
    y = pd.merge(sac_pd, pat_pd, left_on = "subject_id", right_on = "Subject ID")["MS"].values

    if verbose:
        print(f"Number of Samples : {X.shape[0]}")
        print(f"{sac_pd.subject_id.dtype}. {pat_pd['Subject ID'].dtype}. {pat_pd['MS'].dtype}")
        print(f"Shape of the Labels: {y.shape}")   
    
    # Build Group based Test and Train split
    X_train, X_test, y_train, y_test, subid_train, subid_test = validation.grouped_train_test_split(X, y, subject_id, test_fraction = test_fraction) 
   
    # Perform a Reshape 
    X_train = np.reshape(X_train, newshape = (-1, 1))
    X_test = np.reshape(X_test, newshape = (-1, 1))

    # Build a model and fit it
    log_model = LogisticRegressionCV(cv=5)
    log_model.fit(X_train, y_train)
    logistic_results = log_model.predict(X_test)
    acc = accuracy_score(y_test, logistic_results)
    print(f"Test Accuracy is: {acc:.2f}")  

def run_saccade_regression(file_dir, csv_file, patients_file, test_fraction = 0.1, amplitude = 0.0, num_samples = 1150, append = False, verbose = False):
    """
    Build the saccades csv file and then run logistic regression on it. 
    @param file_dir:  
    """
    build_saccades_csv(file_dir = file_dir, 
                       csv_file = csv_file, 
                       amplitude = amplitude,
                       num_samples = num_samples,
                       append = append,
                       verbose = verbose,
                       plot_trace = False)

    # Run Logistic regression and print the output
    run_logistic_regression(csv_file = csv_file, patients_file = patients_file, test_fraction = test_fraction, verbose = verbose)
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Add arguments
    parser.add_argument("--file_dir", type = str, default = "/home/shrikant/EnVision/tsc/mat/patient_mat",
                        help = "Path to the directory which contains the traces")
    parser.add_argument("--csv_file", type = str, default = "./labelled_saccades_frequency.csv",
                        help = "File Name to which micro-saccade frequency should be saved")
    parser.add_argument("--amplitude", type = float, default = 0.0,
                        help = "Minimum Amplitude to classify as a saccade")
    parser.add_argument("--num_samples", type = int, default = 1150, 
                        help = "Number of Samples to downsample to.")
    parser.add_argument("--verbose", type = str, default = "false", 
                        help = "Prints additional information during processing.")
    parser.add_argument("--append", type = str, default = "false",
                        help = "Append the micro-saccades frequency to the csv.")
    parser.add_argument("--patients_file", type = str, default = "/home/shrikant/EnVision/data/updated_patients_stats.csv",
                        help = "Path to the file that contains all the patient information for the trial")
    parser.add_argument("--test_fraction", type = float, default = 0.1,
                        help = "The fraction of dataset that should be used for testing.")
    FLAGS = parser.parse_args()
    append = False 
    verbose = False
    
    if FLAGS.append.lower() == "true":
        append = True
    if FLAGS.verbose.lower() == "true":
        verbose = True
    
    # Count Saccades and then run logisitc Regression
    run_saccade_regression(file_dir = FLAGS.file_dir,
                           csv_file = FLAGS.csv_file,
                           amplitude = FLAGS.amplitude,
                           num_samples = FLAGS.num_samples,
                           patients_file = FLAGS.patients_file,
                           test_fraction = FLAGS.test_fraction,
                           append = append,
                           verbose = verbose)
