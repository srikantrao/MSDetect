"""
Count the number of saccades in the Pupil trace.
Provide some additional information along with that as well.
"""

import numpy as np
import pandas as pd
import streamlit as st
import glob 
import os 
import matplotlib.pyplot as plt 
import re
import scipy.signal
import pre_processing
import plot_utils
import argparse

def count_saccades(file_path, num_samples = 4600, 
                   amplitude = 0.1, verbose = False):
    """
    Count the Number of X and Y micro-saccades.
    @file_path: Path to the numpy file.
    @num_samples: Number of samples to downsample to. 
    """
    # Get the Head and the Tail of the Path
    head, tail = os.path.split(file_path)
    name = tail.split("_")
    
    # Load the trace 
    trace = np.load(file_path)
    trace = trace[:, 100: 4700]

    # Downsample if num_samples is not None 
    if num_samples < 4600:
        trace = pre_processing.downsample_traces(trace, num_samples, axis = 1)
        if verbose:
            print(f"Trace was down sampled to {trace.shape[1]}")
    
    # Convert Numpy Array to a DataFrame
    df = pd.DataFrame.from_records(trace)
    df = df.transpose()
    df.columns = ['X', 'Y', 'Velocity']
    # Create a Dataframe to keep track of the differences 
    derv = df.diff()

    values = derv[abs(derv) > amplitude].count()
    
    if verbose:
        print(values[0])

    return values[0]


def count_saccades_in_dir(file_dir, amplitude = 0.2,
                          num_samples = 4600, verbose = False):
    """
    Count the number of X Saccades in every file

    """
    files = glob.glob(os.path.join(file_dir, "*10007*.npy"))
    num_saccades_list = []
    for fi in files:
        num_saccades_list.append((fi, count_saccades(fi, amplitude = amplitude, num_samples = num_samples, verbose = verbose)))
    
    return num_saccades_list

def build_csv(num_saccades_list, csv_file, append=False, verbose = False):
    """
    Build the csv file which can be used for comparison 
    with hand labelled data.
    @param num_saccades_list - Tuple --> (filename, number of saccades)
    @return: None. Build a csv file and writ
    """
    if append:
        write_mode = 'w+'
    else:
        write_mode = "w"

    with open(csv_file, write_mode) as f:
        f.write("subject_id,eye,version,num_traces\n") 

        for file_name, num_saccades in num_saccades_list:
            file_name = os.path.basename(file_name)
            
            print(file_name)
            sub_id, ver, eye = file_name.split("_")[0:3]

            line = [sub_id] + ["1" if eye == "L" else "2"] + [ver]
            line.append(str(num_saccades))
            f.write(",".join(line) + "\n")


def build_saccades_csv(file_dir, csv_file, amplitude = 0.2, num_samples = 4600, append = False, verbose = False):
    """
    Build a csv which holds information about the number of saccades in each     trace.
    @file_dir: the directory which contains all the traces
    @csv_file: Path of the new csv file that needs to be created 
    """
    
    num_saccades_list = count_saccades_in_dir(file_dir, amplitude = amplitude, num_samples = num_samples, verbose = verbose)

    build_csv(num_saccades_list, csv_file, append = append, verbose = verbose)
    
    if verbose:
        print(f"Micro Saccade frequency has been written to {csv_file}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # File Directory 
    parser.add_argument("--file_dir", type = str, default = "./mat/patient_mat",
                        help = "Path to the directory which contains the traces")
    parser.add_argument("--csv_file", type = str, default = "./automated_saccades_frequency.csv",
                        help = "File Name to which micro-saccade frequency should be saved")
    parser.add_argument("--amplitude", type = float, default = 0.2,
                        help = "Minimum Amplitude to classify as a saccade")
    parser.add_argument("--num_samples", type = int, default = 4600, 
                        help = "Number of Samples to downsample to.")
    parser.add_argument("--verbose", type = str, default = "false", 
                        help = "Prints additional information during processing.")
    parser.add_argument("--append", type = bool, default = False,
                        help = "Append the micro-saccades frequency to the csv.")
    FLAGS = parser.parse_args()
    
    verbose = False
    if FLAGS.verbose.lower() == "true":
        verbose = True

    # Run
    build_saccades_csv(file_path = FLAGS.file_dir, 
                       csv_file = FLAGS.csv_file, 
                       amplitude = FLAGS.amplitude,
                       num_samples = FLAGS.num_samples,
                       append = FLAGS.append,
                       verbose = verbose)
