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
    def ms_infer(self, model_name = "./model/resnet_overfit_6layer.h5",
                 input_file = "/data/envision_working_traces/20107r_V002_meanrem_480_hz_7738.matvelocity.fig.mat",
                 patient_file_path="/home/shrikant/EnVision/data/patient_stats.csv",
                 plot_trace=True,
                 verbose=True):

        st.write("Running Inference on Pupil traces ")

        # Load the Model
        inference_model = load_model(str(model_name))

        if verbose:
            inference_model.summary()

        # Load the Input trace
        trace = inference_utils.read_single_trace(input_file,
                                                  patient_file_path,
                                                  start_index = 10)
        if verbose:
            st.write(f"Matrix file has been loaded.")
            st.write(f"Shape of the input trace is:{trace.xraw.shape[0]}")
        # Get Hold of all the features used
        X, y, subject_ids = pre_processing.traces_to_feature([trace], velocity=False, mean_center=True, scale_std=True)
        
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
