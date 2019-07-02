"""
Quick Script to compare the output of the auto generated micro-saccades to the 
labelled micro-saccades 
"""

import numpy as np
import pandas as pd 
import streamlit as st 
import argparse 

def build_dict(file_path, verbose = True):
    """
    Build a dictionary with the csv
    @param file_path: Path to the csv file  
    @param verbose: Print some additional information
    """
    with open(file_path, 'r') as f:
        # Read the first line as it is just columns 
        f.readline()       
        saccade_dict = {}
        for line in f.readlines():
            line = line.strip().split(",")
            key = line[0] + "_" + line[1] + "_" + line[2] 
            saccade_dict[key] = int(line[3])

    return saccade_dict

def compare_dicts(label_dict, auto_dict, verbose = True):
    """
    Compare the two dicts
    """
    error_dict = {}

    for key, value in label_dict.items():
        if key in auto_dict:
            error_dict[key] = label_dict[key] - auto_dict[key]
        
    return error_dict

def write_csv(error_dict, csv_file):
    """
    Write the values of the Error dict to the 
    @param error_dict: The Error Dict 
    @param csv_file: Path to the which the csv file should be written 
    @return : None
    """
    with open(csv_file, 'w') as f:
        f.write("subject_id,eye,version,num_traces\n")
        
        for key, value in error_dict.items():
            line = key.split("_")
            line = line + [str(value)]
            f.write(",".join(line) + "\n")

def compare_csvs(label_csv, auto_csv, error_csv):
    """
    Compare the labelled csv and the auto generated csv 
    @param label_csv: Path to the labelled csv 
    @param auto_csv: Path to the auto generated csv 
    @param error_csv: Path to the error csv 
    """

    label_dict = build_dict(label_csv)
    auto_dict = build_dict(auto_csv)
    error_dict = compare_dicts(label_dict, auto_dict)
    write_csv(error_dict, error_csv)
       

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--label_csv", type = str, 
                        default = "../../data/labelled_saccades_frequency.csv",
                        help = "Path to the labelled csv. ")
    parser.add_argument("--auto_csv", type = str, 
                        default = "../../data/automated_saccades_frequency.csv",
                        help = "Path to the auto generated csv")
    parser.add_argument("--error_csv", type = str, 
                        default = "../../data/error.csv",
                        help = "Path to the Error csv file")
    FLAGS = parser.parse_args()
    
    compare_csvs(FLAGS.label_csv, FLAGS.auto_csv, FLAGS.error_csv)
