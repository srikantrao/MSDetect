"""
Build out the spectrogram
Show effects of downsampling 
"""

import numpy as np
import pandas as pd 
import streamlit as st
import os
import matplotlib.pyplot as plt 
from utils import pre_processing 
from utils import data_manipulation
from utils import validation
from utils import inference_utils
from utils import plot_utils
"""
### What does a spectrogram look like ? 
"""

# Get hold of a trace 
# trace = pre_processing.
# Build a spectrogram - How different does it look when you down sample ? 


# File path of which you want to calculate fft 
file_path = os.environ["TEST_ARR"]

trace = np.load(file_path)
st.write(trace[0].shape)

# Calculate and plot spectrogram for a wav audio file
def graph_spectrogram(data):
    
    nfft = 256 # Length of each window segment
    fs = 480 # Sampling frequencies
    noverlap = 128 # Overlap between windows
    nchannels = data.ndim
    
    if nchannels == 1:
        pxx, freqs, bins, im = plt.specgram(data, nfft, fs, noverlap = noverlap)
    elif nchannels == 2:
        pxx, freqs, bins, im = plt.specgram(data[:,0], nfft, fs, noverlap = noverlap)
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.title("Spectrogram of Horizontal motion of Pupil trace")
    st.pyplot()
    return pxx

graph_spectrogram(trace[0])
