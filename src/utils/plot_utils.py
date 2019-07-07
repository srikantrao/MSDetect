""" 
Plotting util functions for EDA and inference
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import streamlit as st 
import glob 
import os

def plotXY(array, xlabel, ylabel, use_streamlit, title="Simple Plot"):
    """
    Plot the XY pupil trace
    @param X: Input numpy array that contains the features that
              need to be plotted.
    @param xlabel: X axis label
    @param ylabel: Y axis label
    @param use_streamlit: Whether streamlit should be used or not
    returns: None. Uses Matplotlib and Streamlit to plot 
    """
    plt.figure()
    plt.plot(array[0, :, 0], label="X Trace")
    plt.plot(array[0, :, 1], label="Y Trace")
    plt.legend(loc="upper left")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if use_streamlit:
        st.pyplot()


def plot_random_traces(array, nrows, ncols, xlabel="X axis", 
                       ylabel="Y axis", title="Simple Plot"):
    """
    Use to plot X and Y traces for multiple samples
    @param nrows - Number of rows 
    @param ncols - Number of cols
    @param xlabel - X axis label
    @param ylabel - Y axis label
    @param title - Title of the 
    """

    indices = np.random.randint(0, array.shape[0], size=(ncols * nrows))

    fig, ax = plt.subplot(nrows, ncols)

    plt.figure()
    
    for i, pl in enumerate(ax.flatten()):
        X_val = X[indices[i], 0]
        Y_val = X[indices[i], 1]
        pl.plot(time, X_val, label="X Position")
        pl.plot(time, Y_val, label="Y Position")
        pl.legend(loc='upper left')
        pl.set_xlabel(xlabel)
        pl.set_ylabel(ylabel)
        pl.set_title(f"{title}:{i}")

    plt. tight_layout()

def plot_histogram(data, title="title", xlabel="X axis", ylabel="Y Axis",
                   density=False, hist_range = None):
    """
    Using Object Oriented API of matplotlib to plot plot_histogram
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    N, bins, patches = ax.hist(data, edgecolor='white', linewidth=1,
                               density = density, stacked = True, 
                               range = hist_range, color = "orange")
    
    # Set the Labels and the titles 
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    # Set grid as the background 
    ax.set_axisbelow(True)
    ax.yaxis.grid(color = 'gray', linestyle = 'dashed')
    ax.xaxis.grid(color = 'gray', linestyle = 'dashed')
    return fig  

def plot_line(y_data, x_data = None, title = "Title",
              xlabel = "X axis", ylabel = "Y axis"):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    if not x_data:
        plt.plot(y_data, color='r', marker='o')
    else:
        plt.plot(x_data, y_data, color = 'r')
        plt.plot(x_data, y_data, 'bo')

    # Set the Labels and the titles 
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    # Set grid as the background 
    ax.set_axisbelow(True)
    ax.yaxis.grid(color = 'gray', linestyle = 'dashed')
    ax.xaxis.grid(color = 'gray', linestyle = 'dashed')
    
    return fig  
    
def set_plot_properties():
    
    fig, ax = plt.subplots() 
    
    # Set grid as the background 
    ax.set_axisbelow(True)
    ax.yaxis.grid(color = 'gray', linestyle = 'dashed')
    ax.xaxis.grid(color = 'gray', linestyle = 'dashed')
    
    return fig, ax  

def plot_all_traces(file_path, subject_id):
    """
    Plot all the traces for that particular subject ID.
    @param file_path: File path that contains all the matrices
    @param subject_id: Subject ID 
    @return: None. Plotting objects get created. 
     """
    filt_files = glob.glob(os.path.join(file_path, f"*{subject_id}*.npy"))
    # Plot the files now
    n_cols = 2
    if len(filt_files) % 2 == 0:
        n_rows = len(filt_files) // n_cols
    else:
        n_rows = len(filt_files) // n_cols + 1

    fig, ax = plt.subplots(n_rows, n_cols, figsize=(16, 24))

    for i, pl in enumerate(ax.flatten()):
        if i >= len(filt_files):
            continue
        array = np.load(filt_files[i])
        pl.plot(array[0, 100:4700], label="X trace")
        pl.plot(array[1, 100:4700], label="Y trace")
        pl.legend(loc="upper right")
        pl.set_title(filt_files[i].split("/")[-1])
        # Set grid as the background 
        pl.set_axisbelow(True)
        pl.yaxis.grid(color = 'gray', linestyle = 'dashed')
        pl.xaxis.grid(color = 'gray', linestyle = 'dashed')
        pl.set_xlabel("Time Series")
        pl.set_ylabel("Amplitude Degrees")
    return fig  
