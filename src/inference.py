"""
Script to run inference using on single test cases.
Pre-trained (TF or Keras) model is loaded and inference is run.
"""

# Import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import scipy.signal
import argparse
import fire
import os
import sys
import glob 
from keras.models import load_model
from utils import data_manipulation
from utils import pre_processing
from utils import inference_utils 
from utils import plot_utils

# Show only Error logs for Tensorflow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
np.set_printoptions(precision = 2)

class Inference:
    """
    Running inference on the pupil trace of a patient. 
    Plots the trace and provides prediction based on provided threshold.
    """
    def ms_infer(self, 
                 model_name = "INF_MODEL",
                 input_file = "20107r_V002_meanrem_480_hz_7738.matvelocity.fig.mat",
                 patient_file_path="PATSTAT",
                 plot_trace=True,
                 verbose=True):
        
        # Generate Directory Paths from Env Variables  
        patient_file_path = os.environ[patient_file_path]
        model_name = os.environ[model_name]
        st.write("## Running Inference on Pupil traces ")
        
        # Building a list of Patient IDs to choose from 
        patient_list, pat_str_list = inference_utils.load_patient_ids(patient_file_path)
        
        index = st.selectbox(label = "Select the Patient ID",
                             options = pat_str_list)
        
        # Get hold of all the files of that particular subject 
        file_list = inference_utils.get_file_list(patient_list[index])

        # Find version of the traces available
        ver_list = inference_utils.get_ver_list(file_list)
        if not ver_list:
            st.error(f"No traces for this {patient_list[index]}")
            return 

        # Select the version of the  trace 
        ver_index = st.selectbox(label = f"Select trace version for {patient_list[index]}",
                                 options = ver_list)
        file_list = inference_utils.filter_by_version(file_list, ver_list[ver_index])

        # Find if both right eye and left eye are available 
        eye_list, eye_str_list = inference_utils.get_eye_list(file_list) 

        if not eye_list:
            st.error(f"No traces for this {patient_list[index]} and {ver_index}")
        # Select the version of the eyes available 
        eye_index = st.selectbox(label = "Select Right or Left Eye",
                                 options = eye_str_list)
        
        # Filter the list by the eye that was selected
        file_list = inference_utils.filter_by_eye(file_list, eye_list[eye_index])
        st.write(f"Inference is being run on **Patient ID: {patient_list[index]}** on **{ver_list[ver_index]}** and on the **{eye_str_list[eye_index].lower()}** trace")
             
        # Add the play button here. Maybe that might be better 
        if st.button("Play"):
            # Load the Model
            inference_model = load_model(model_name)
 
            if verbose:
                inference_model.summary()
        
            # Load the Input trace
            input_file = file_list[0]
            trace = inference_utils.read_single_trace(input_file,
                                                      patient_file_path,
                                                      start_index = 10)
            if verbose:
                st.write(f"Pupil trace has been loaded.")
            # Get Hold of all the features used
            X, y, subject_ids = pre_processing.traces_to_feature([trace], velocity = False, mean_center = False, scale_std = False)
        
            # Make inputs channel last 
            X = data_manipulation.channel_last(X)
            y = y.astype(np.int32).squeeze()
        
            if plot_trace:
                plot_utils.plotXY(X, "Time Steps", "Position", use_streamlit=True, title="X and Y Pupil traces")

            # Predict the label
            y_pred = inference_model.predict(X)

            # Predict the output of the trace
            y_est = np.argmax(y_pred)

            if y_est == y:
                st.write(f"Correct prediction. Label : {y}. Predicted: {y_est}")
            else:
                st.write(f"Wrong Prediciton. Label : {y}. Predicted: {y_est}")
       
            # Print out additional information
            if verbose:
                st.write(f"Probability that the patient does not have MS: {y_pred[0,0]:.2f}")
                st.write(f"Probability that the patient has M.S: {y_pred[0,1]:.2f}")
                if y == 1:
                    st.write(f"EDSS Score is: {trace.sub_edss}")
        

if __name__ == "__main__":
    
    fire.Fire(Inference)

