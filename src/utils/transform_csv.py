"""
Build a script to calculate the number of Saccades in each trace  
"""
import numpy as np
import pandas as pd 
import streamlit as st 
from collections import defaultdict
import sys 
import os 
import glob 
import argparse 


def transform_csv(file_path, csv_file):

    # Create Dict. 
    # Key = id_eye_tracenum. Value = [num_saccades, [x1, x2]]
    subject_dict = defaultdict(lambda: [0, []])

    with open(file_path, 'r') as f:
        # Get rid of the first line since that contains the headers 
        line = f.readline()
    
        for i, line in enumerate(f.readlines()):
            line = line.strip().split(",")
            key = f"{line[0]}_{line[1]}_{line[2]}"
            subject_dict[key][0] += 1
            subject_dict[key][1].append(line[9])

    # Write this to a csv file
    with open(csv_file, 'w') as f:
        # Write all the headers 
        f.write("subject_id,eye,version,num_traces\n")
        line = []
        for key, value in subject_dict.items():
            # Values stored in key
            line = key.split("_")
            # Values stored in value 
            line.append(str(value[0]))
            line = line + value[1]
            f.write(",".join(line) + "\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type = str, default = "./patients_updated3.12.19_acc50k_allmicro.csv", 
                        help = "Path to the hand-labelled csv file")
    parser.add_argument("--csv_file", type = str, default = "./saccades_frequency.csv", 
                        help = "File name to which saccade frequency should be saved in csv format.")

    FLAGS = parser.parse_args()

    # Run
    transform_csv(FLAGS.file_path, FLAGS.csv_file)
    
