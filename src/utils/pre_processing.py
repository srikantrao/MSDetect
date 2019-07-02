""" Contains all the methods that are used to perform pre-processing steps on the XY traces 
    Author - Srikant Rao
"""
# Import Statements 
import os 
import numpy as np
import pandas as pd
import csv
import logging 
import scipy.signal
from typing import List
from utils.data_types import Xygroup
from collections import defaultdict

logging = logging.getLogger(__name__)

def downsample_by(X: np.ndarray, ds_ratio: float, axis: int = 2) -> np.ndarray:
    """
    Downsamples X by ds_ratio in the axis that has been specified 
    @ param X: Input numpy array 
    @ param ds_ratio: Ratio to be used
    @ axis: The axis along which this should be performed. Default = 2
    @ return: Numpy array which is a downsampled output of X.
    """
    num_samples = X.shape[axis] // ds_ratio
    return downsampled_traces(X, num_samples, axis)

def spec_build(array, nfft=128, window_size = 100, window_stride = 50):
    """
    Build a spectrogram for the input array. Wrapper function which uses scipy.signal.spectrogram to generate the spectrogram
    @param array: Input numpy array. Should contain X and Y traces.
    @param nfft: Number of FFT samples 
    @param window_size: Number of samples in each STFT window
    @param window_stride: Stride of each sample size.
    @returns: Numpy array containing the spectrogram.abs
    """
    overlap = window_size - window_stride
    f, t, Sxx = scipy.signal.spectrogram(array, nfft = nfft,
                                         window = ('tukey', window_size),
                                         noverlap = overlap,
                                         nperseg=window_size,
                                         mode = 'magnitude')
    return Sxx

def downsample_traces(X: np.ndarray, num_samples: int, axis = 2) -> np.ndarray:
    """
    Uses Scipy Resampling technique to downsample trace data along axis = 2 
    to num_samples samples.
    Resample x to num samples using Fourier method along the given axis.
    The resampled signal starts at the same value as x but is sampled with a 
    spacing of len(x) / num * (spacing of x). 
    Because a Fourier method is used, the signal is assumed to be periodic.
    
    @param X: Numpy Array on which downsampling needs to be performed. 
    @param num_samples: Number of samples in the final output.
    @ param axis: Axis along which the downsampling should be performed.
    @ return: Numpy array downsampled to num_samples 
    """
    return scipy.signal.resample(X, num_samples, axis=axis)

def add_spatial_blurring(X: np.ndarray, stddev: float) -> np.ndarray:
    """
    Adds Gaussian Noise to X, the input numpy array with mean =0 and stddev 
    set by stddev.
    @ param X: Input Numpy Array 
    @ param stddev: Standard Deviation to used for the Gaussian Blurring.
    @ return: Input numpy array with gaussian noise added.
    """
    input_shape = X.shape

    noise = np.random.normal(scale=stddev, shape=input_shape)

    return X + noise

def build_using_edss(traces, use_velociy=False,
                     mean_center = False,
                     scale_std=False):
    """
    Pre-process the traces to include only using edss scores.
    @param traces: A list of Eyetrace objects parsed from the data directory. 
    @use_velocity: If True, velocity is added as a feature to the trace.
    @mean_center: If True, the array is centered around the mean of each trace.
    @scale_std: If True, the array is scaled by the standard deviatino of each trace. Provides ability with mean_center to 
                convert to normal form.
    @returns: numpy array of input features which will contains atleast X and Y positions. Numpy array of the labels for each trace as well 
              subject id.
    """
    # Filter out all the Control traces.
    traces = list(filter(lambda t: t.sub_ms == '1', traces))

    X, y, sub_id = traces_to_feature(traces, velocity=use_velociy,
                                     mean_center=mean_center,
                                     scale_std=scale_std,
                                     key= lambda t: float(t.sub_edss) > 3)

    return X, y, sub_id


def traces_to_feature(
    trials: List["EyeTrace"],
    position: bool = True,
    velocity: bool = False,
    saccades: bool = False,
    mean_center: bool = False,
    scale_std: bool = False,
    key = lambda t: t.is_patient
) -> Xygroup:
    """ 
    Convert a list of EyeTrace objects to numpy arrays.
    @param: trials - List of traces
    @param: mean_center - Whether or not to subtract the mean from the raw trace data
    @param: scale_std - Whether or not to scale the raw trace data by its standard deviation
    @param: filter_key - Key to be used to categorize entries into 1 and 0. 
    @returns: X - Numpy array of features, y
             y - Numpy array of labels
             subject_ids - Subject ID
    """
    # Add the X and Y position
    if position: 
        X = np.array([[t.xraw, t.yraw] for t in trials])
        print(f"X Shape is {X.shape}")
    
    # Add the raw velocity 
    if velocity:
        vel = np.array([[t.vraw] for t in trials])
        print(f"vel shape is {vel.shape}")
        X = np.concatenate((X, vel), axis=1)
    
    y = np.array([key(t)  for t in trials])
    subject_ids = np.array([t.subjid for t in trials])
    if mean_center:
        X -= X.mean(axis=2, keepdims=True)
    if scale_std:
        X /= X.std(axis=2, keepdims=True)
    return X, y, subject_ids

def flip_augment(X, y, sub_id, indices = None):
    """
    Augment the dataset by flipping X and Y for particular indices
    @param X: numpy array which contains the features of the given dataset. 
    @param y: numpy array which contains the labels corresponding to the dataset
    @param indices: Indices of X which should undergo flipping. If None, all entries in X are flipped, else only particular indices. Default = None.  
    @return: Numpy array for augmented X and y as well the corresponding subject ID.
    """
    if indices is None:
        X_aug, y_aug, sub_id_aug = -X, y, sub_id 
    else:
        X_aug, y_aug = -X[indices], y[indices]
        sub_id_aug = sub_id[indices]

    return X_aug, y_aug, sub_id_aug

def load_csv_data(filepath):
    """
    Load csv data into a pandas dataframe.
    @param filepath: Path to the file 
    @returns: df, A Pandas dataframe
    """
    df = pd.read_csv(filepath)
    return df

def use_dist(array):
    """
    Use this to combine the X and Y features to use only the Distance as a feature to the  
    @param - Numpy array which contains both X and Y traces.
    @ returns - Numpy array which contains distance
    """
    X = np.sqrt(np.power(array[:, 0, :], 2) + np.power(array[: , 1, :], 2))
    return np.reshape(X, newshape = (X.shape[0], 1, X.shape[1]))

def build_dict(csv_path, row = 10):
    """
    Build a Dict. Key = Subject ID. Value = List of Micro-saccades observed in readings
    @csv_path: Path to the csv file
    @row: Particular column in the csv file that contains the information that is needed.
    """
    # Use defaultdict for tolerance to ValueError 
    subject_dict = defaultdict(list)

    with open(csv_path, 'r') as f:
        for index, line in enumerate(f.readlines()):
            if index == 0:
                continue
            line = line.strip().split(",")
            sub_id = int(line[1])
            saccades = int(line[row])
            subject_dict[sub_id].append(saccades)
    
    return subject_dict

def in_subject_mean_std(subject_dict):
    """
    Calculate the in subject meand and variance 
    for all subjects
    @subject_dict: Dict with Key = Subject ID. Value = List of micro-
                   saccade values per trace
    """
    # Use this to determine the in subject variation
    return {key : (np.mean(value), np.std(value))for key, value in subject_dict.items()}    

def build_dataFrame_from_csv(patient_file, control_file, save_file_path=None):
    """
    Build a pandas dataframe which stores the information of micro-saccades per subject
    @patient_file: Path to the patient file 
    @control_file: Path to the control file 
    """
    pass
