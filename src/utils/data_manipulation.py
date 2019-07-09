"""
Functions for reading in and transforming the trace data. Includes:

- data augmentations
- feature generation
- data conversion
"""

import copy
from glob import glob
import logging
import os
import re
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.io import loadmat
import scipy.signal
from sklearn.decomposition import PCA

from utils.data_types import Xygroup

logger = logging.getLogger(__name__)


def load_augmented_trials(
    data_dir: str,
    patient_file_path: str,
    filter_age: Optional[int] = None,
    flip_y: bool = False,
    num_splits: int = 1,
    trace_start_index: int = 10,
    just_patients: bool = False,
) -> List["EyeTrace"]:
    """
    Load and optionally augment trial data.

    Parameters
    ----------
    data_dir
        Directory containing the trial data
    patient_file_path
        CSV file containing patient data
    filter_age
        If included, then only return patients with ages below filter_age
    flip_y
        Whether or not to double the trial data by flipping the y traces.
    num_splits
        How many times to split up each trial.
    trace_start_index
        The index that will be the start of the trace data. The traces have some
        anomalies from interpolation at the first couple data points, so it's
        recommended to drop those points.
    just_patients
        Whether or not to _only_ return patient data (and not control data).

    Returns
    -------
        trials
            List of trial data after augmentation and filtering

    """
    traces = readin_traces(
        data_dir, patient_file_path, start_index=trace_start_index
    )

    if just_patients:
        traces = [t for t in traces if t.is_patient]

    if filter_age is not None:
        traces = [t for t in traces if int(t.sub_age.item()) < filter_age]

    # flip_y
    trials = []
    for trace in traces:
        trials.append(trace)
        if flip_y:
            new_trace = copy.deepcopy(trace)
            new_trace.y = -new_trace.y
            trials.append(new_trace)

    # split trials
    trials = split_trials(trials, num_splits)
    return trials


def channel_last(X: np.ndarray) -> np.ndarray:
    """
    Change the shape of the trace matrix to have the channel as the last
    dimension
    """
    X_channel = X[:, 0, :]
    if X.shape[1] == 1:
        return X_channel 
    else:
        Y_channel = X[:, 1, :]
        X = np.stack((X_channel, Y_channel), axis=2)
    return X


def one_hot_encode_y(y: np.ndarray) -> np.ndarray:
    """
    Convert a boolean vector y to a one-hot-encoded matrix:

    Example
    -------

    Converts

    [False, True, True, False]

    to

    [[1, 0],
     [0, 1],
     [0, 1],
     [1, 0]]
    """
    y_new = np.zeros((len(y), 2))
    y_new[y, 1] = 1
    y_new[~y, 0] = 1
    return y_new


def traces_to_arrays(
    trials: List["EyeTrace"],
    mean_center: bool = False,
    scale_std: bool = False,
    key = lambda t: t.is_patient
) -> Xygroup:
    """
    Convert a list of traces to arrays that are amenable with scikit-learn.

    Parameters
    ----------
    trials
        List of traces
    mean_center
        Whether or not to subtract the mean from the raw trace data
    scale_std
        Whether or not to scale the raw trace data by its standard deviation
    filter_key
        Key to be used to categorize entries into 1 and 0
    Returns
    -------
    X
        "Feature" matrix consisting of the x and y raw trace data. The shape is
        number of traces, channel, time. "Channel" means x vs. y. We use the
        term channel so that we can think of this like an RGB channel when
        passing this data through convolutions.
    y
        Boolean array corresponding to whether or not the subject is a patient.
    subject_ids
        Array of strings indicating which subject the trace data belongs to.
        This should be used as the "group" array for all of the grouped cross
        validation routines

    """
    # shape = num_traces, (x, y), time
    X = np.array([[t.xraw, t.yraw] for t in trials])
    y = np.array([key(t)  for t in trials])
    subject_ids = np.array([t.subjid for t in trials])
    if mean_center:
        X -= X.mean(axis=2, keepdims=True)
    if scale_std:
        X /= X.std(axis=2, keepdims=True)
    return X, y, subject_ids


def split_trace_data(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, num_splits: int
) -> Xygroup:
    """
    Split the trace data into num_splits copies. This is used for data
    augmentation to artificially increase the size of the training dataset.
    """
    total_num_splits = X.shape[0] * num_splits
    split_size = X.shape[2] // num_splits
    if X.shape[2] % num_splits != 0:
        logger.warning(
            f"Dataset size({X.shape[2]}) is not divisible "
            f"by number of splits {num_splits}. Remaining data "
            f"will be dropped."
        )
    out_X = np.zeros((total_num_splits, X.shape[1], split_size))
    out_y = np.zeros((total_num_splits,), dtype=np.bool)
    out_groups = np.empty((total_num_splits,), dtype=groups.dtype)

    for i in range(num_splits):
        out_X[i * X.shape[0]: (i + 1) * X.shape[0], ...] = X[
            ..., i * split_size: (i + 1) * split_size
        ]
        out_y[i * X.shape[0]: (i + 1) * X.shape[0]] = y[:]
        out_groups[i * X.shape[0] : (i + 1) * X.shape[0]] = groups[:]

    return out_X, out_y, out_groups


def flip_y_traces(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> Xygroup:
    """
    Double the training data by creating duplicate data with flipped y traces.
    """
    X_flip = X.copy()
    X_flip[:, 1, :] = -X_flip[:, 1, :]
    return (
        np.vstack((X, X_flip)),
        np.concatenate((y, y)),
        np.concatenate((groups, groups)),
    )


def get_spectrograms(
    traces: List["EyeTrace"], cutoff_idx: int = 20
) -> np.ndarray:
    """Convert trace data to spectrograms"""
    spectrograms = []
    for t in traces:
        f, t_, Sxx = scipy.signal.spectrogram(t.yraw)
        spectrograms.append(Sxx[:cutoff_idx, :].ravel())
    return np.vstack(spectrograms)

def smooth_traces(
    X: np.ndarray, window_length: int, polyorder: int
) -> np.ndarray:
    """
    Smooth trace data via a savgol_filter. It is assumed that the time dimension
    is axis=2.
    """
    X = scipy.signal.savgol_filter(X, window_length, polyorder, axis=2)
    return X


def downsample_traces(X: np.ndarray, num_samples: int) -> np.ndarray:
    """
    Downsample trace data along the time axis (2) to num_samples samples.
    """
    return scipy.signal.resample(X, num_samples, axis=2)


class EyeTrace:
    """A class to contain eye trace information
    Parameters:
        xlocs (numpy array) set of x positions of eye
        ylocs (numpy array) set of y positions of eye
        time (numpy array) set of timepoints in register with xlocs and ylocs
        velocities (numpy array) set of velocities of eye *IGNORED RIGHT NOW**))
        interptimes (numpy array) set of timepoints that have been interpolated
        bts (numpy array) set of timepoints that have blinks
        nbs (numpy array) number of blinks
        ppd (float) Number of pixels per degrees of eyetraces
        subinfo (Pandas DF) line from pandas df contining info about patients
    Returns:
        Instance of Class eyetrace
    """

    def __init__(
        self,
        xlocs,
        ylocs,
        time,
        velocities,
        interptimes,
        version,
        bts,
        bns,
        ppd,
        fname,
        subinfo,
    ):
        # define filename
        self.fname_full = fname
        self.fname_root = os.path.split(fname)[0]
        self.fname = os.path.split(fname)[-1]
        self.version = version
        # not all filenames have 'meanrem' in name, but first 2 are always consistent
        # subjideye_tracenum_meanrem_480hz_####..mat
        split = self.fname.split("_")
        self.subjid = re.search(r"\d+", self.fname)[0]
        self.eye = split[0][-1].upper()  # 'l' or 'r' for left/right eye
        self.tracenum = split[1]  # v001, v002, v003, v004
        self.traceid = self.fname.replace(
            "..mat", ""
        )  # some filenames have two dots..
        # self.traceid = self.traceid.replace('.mat*.mat', '') # some filenames have this format

        # define scaled time
        self.time_offset = time[0]
        self.time = time - self.time_offset
        # round to 10 decimal places to avoid inconsistencies
        self.time = np.round_(self.time, 10)

        # define dt
        self.dt = time[1] - time[0]
        # define ppd value
        self.ppd = ppd

        # define x and y position attributes
        # raw values: center at start
        self.xraw = xlocs
        self.yraw = ylocs
        # calcualte velocities
        self.xvel = self.xraw[1:] - self.xraw[:-1]
        self.yvel = self.yraw[1:] - self.yraw[:-1]
        self.vraw = (
            np.sqrt(np.square(self.xvel) + np.square(self.yvel)) / self.dt
        )
        self.vraw = np.append(self.vraw, self.vraw[-1])  # same length

        # define timepoints as saccades (not saccade=drift)
        self.sac_idx = self.vraw > 10  # 10 degrees per second

        # define times that have been interpolated
        self.interp_times = interptimes - self.time_offset
        # indices of self.time where any of self.interp_times is in self.time within tol
        self.interp_idx = self.in1d_close(
            self.interp_times, self.time, tol=1e-6
        )  # True indicates that there was interpolation

        # define number of blinks and times
        self.blink_times = bts
        self.blink_num = bns

        # values that are changeable
        self.x = np.subtract(xlocs, xlocs[0])
        self.y = np.subtract(ylocs, ylocs[0])
        self.v = self.vraw

        # DO THIS INSTEAD OUTSIDE READIN
        # get subject info variables from datasheet
        # pf = rut.read_pstats(patientfile)
        self.subinfo = subinfo
        self.is_patient = bool(int(subinfo["MS"].iloc[0]))
        self.sub_ms = np.array(subinfo["MS"]).squeeze()
        self.sub_age = np.array(subinfo["Age"]).squeeze()
        self.sub_edss = np.array(subinfo["EDSS"]).squeeze()
        self.disease_dur = np.array(subinfo["Diseasedur"]).squeeze()
        self.progressive = np.array(subinfo["progressive"]).squeeze()
        self.cerebellar = np.array(subinfo["Cerebellar"]).squeeze()
        self.brainstem = np.array(subinfo["Brainstem"]).squeeze()

    def in1d_close(self, a, b, tol=1e-6):
        """
        Calculates a in b up to some tolerance.
        i.e. Find all values in a that match any value in b within some tolorance
        Parameters:
          a (numpy array) values to look for
          b (numpy array) array to look in
          tol (float) tolerance
        Returns:
          in1d : ndarray, bool
            The values from a that are in b are located at b[in1d].
        Notes:
            adapted from https://github.com/numpy/numpy/issues/7784
            defines inclusion as being [a[i] - tol, a[i] + tol)
            test:
                a = np.array([1, 1.1, 1.3, 1.6, 2.0, 2.5, 3, 3.4, 3.8])
                b = np.linspace(0, 10, 101)
                c = (((b >= 0.8) & (b < 2.2)) |
                     ((b >= 2.3) & (b < 2.7)) |
                     ((b >= 2.8) & (b < 4.0)))
                assert np.all(in1d_close(a, b, tol=0.2) == c)
        """
        a = np.unique(a)
        intervals = np.empty(2 * a.size, float)
        intervals[::2] = a - tol
        intervals[1::2] = a + tol
        overlaps = intervals[:-1] >= intervals[1:]
        overlaps[1:] = overlaps[1:] | overlaps[:-1]
        keep = np.concatenate((~overlaps, [True]))
        intervals = intervals[keep]
        return np.searchsorted(intervals, b, side="right") & 1 == 1


def read_n_filter(filename, subinfo, version,
                  start_index=0, stop_index = 4800):
    """
    Read in our files, filter them, and return an eyetrace instance
    Parameters:
        Filename (str) Location of the .mat file containig the eyetrace
        subinfo (PD array)    line from pd datasheet containing subject information
        start_index (int) Grab data from each trace starting at the start_index
    Returns:
        trial (Eyetrace) Eyetrace instance for a given trial (variable len)
    """
    ppd = 102  # pixels per degree
    # ppc_arcmin = 1. / ppd * 60. # Dont use conversion from pixels to arcmins
    # load file
    nfile = loadmat(filename)
    
    # extract features
    xss = nfile["xmotion"][0][start_index:stop_index]
    yss = nfile["ymotion"][0][start_index:stop_index] 
    tss = nfile["timesecs"][0][start_index:stop_index]
    vss = nfile["velocity"][0][start_index:stop_index]

    # blink info
    bts = nfile["blinktimes"]
    bns = nfile["blinknum"][0]

    # identify bad timepoints but do NOT delete them, they are being interpolated before readin.
    interp_times = nfile["badtimes"][:, 0]

    # convert file contents into EyeTrace object
    trial = EyeTrace(
        xss, yss, tss, vss, interp_times, version, 
        bts, bns, ppd, filename, subinfo)
    return trial


def read_pstats(fpath):
    """
    Read in the Patient Info Datasheet
    Parameters:
        fpath (str) Path to patient info sheet
    Returns:
        pf (pandas df) Formatted Patient Info Datahsheet
    """

    # read in using pandas
    pf = pd.read_csv(fpath, sep=",", dtype=object)

    # remove NAN rows
    pf = pf.dropna(axis=0, how="all", subset={"Subject ID"})

    # keep interesting columns
    """keepcols = [
        "Subject ID",
        "MS",
        "Age",
        "EDSS",
        "IS",
        "RRMS",
        "PP",
        "progressive",
        "Diseasedur",
        "Brainstem",
        "Pyramidal",
        "Cerebellar",
        "Sensory",
    ]"""
    return pf


def readin_traces(dataDir, patientFilePath, start_index=0, 
                  stop_index = 4799):
    """
    read in set of traces
    Parameters:
        Datadir (str) Folder containing .mat files that each have eyetraces.
        patientFilePath (str) filename containing patient information
        start_index (int) Grab data from each trace starting at the start_index
    Returns:
        Trials (list) List of trials, each an eyetrace isntance
    """
    files = [
        y for x in os.walk(dataDir) for y in glob(os.path.join(x[0], "*.mat"))
    ]

    # patient file with info
    pf = read_pstats(patientFilePath)

    # loop through and read in all files, each as a trial.
    pattern = re.compile(r"\d+")
    trials = []
    for fi in files:
        # get subject info variables from datasheet
        fname = os.path.split(fi)[-1]
        match = re.search(pattern, fname)
        if not match:
            print(f"Warning: could not parse subject id for file {fname}")
            continue
        keys = fname.strip().split("_")
        subjid = match[0]
        version = keys[1] + "_" + keys[0][-1]
        subinfo = pf[pf["Subject ID"] == subjid]
        if subinfo.empty:
            print(f"Cant find Subject ID: {subjid}, {fname}")
            continue
        # read in our trial trace
        trial = read_n_filter(fi, subinfo, version, 
                              start_index=start_index,
                              stop_index = stop_index)
        trials.append(trial)
    return trials


def rescale_data_to_one(data):
    """
    Rescales input data to be between 0 and 1
    TODO: Verify that this is actually the right thing to do?
        We want all of the data features to be on the same scale
        Should this normalization be done per sample or across the whole dataset?
    """
    data_min = np.amin(data)
    data_max = np.amax(data)
    return (data - data_min) / (data_max - data_min)


def age_features(trials):
    ages = [int(trial.sub_age.item()) for trial in trials]
    max_age = np.max(ages)
    min_age = np.min(ages)
    ages = np.asarray((ages - min_age) / (max_age - min_age + 1e-10))[:, None]
    ages = rescale_data_to_one(ages)
    ages = 2 * np.round(ages / 2, 1)
    return ages


def pca_whiten_reduce(
    trials,
    n_components_or_expl_variance=0.995,
    rand_state=np.random.RandomState(None),
    pca=None,
):
    """
    Parameters:
        data: [np.ndarray] with first dim indicating batch.
        n_components_or_expl_variance: [float] indicating how many PCs should be used by requiring
            a minimum explained variance.
            100*n_components_or_expl_variance = % variance explained of reduced data
            Alternatively, if n_components_or_expl_variance is >= 1.0,
            then it is treated as an integer equal to the number of components to keep
        rand_state [np.random.RandomState()]
    Returns:
        dimensionality reduced & whitened data
    """
    # Reshape data to 2D if it is more than 2D
    minlen = np.min([np.min((len(trial.x), len(trial.y))) for trial in trials])
    # arrray is indexed by [tracenumber, timepoint, dim] where dim=2 is (x,y)
    trials_array = np.zeros((len(trials), minlen, 2))
    # fill in our array
    for i, trial in enumerate(trials):
        trials_array[i, :, 0] = trial.x[:minlen]
        trials_array[i, :, 1] = trial.y[:minlen]
    data = trials_array
    if data.ndim > 2:
        data = data.reshape(data.shape[0], np.prod(data.shape[1:]))
    if pca is None:
        if n_components_or_expl_variance >= 1.0:
            num_components = int(n_components_or_expl_variance)
        else:
            # Do full dim PCA to figure out how many components to keep
            pca = PCA(
                n_components=None,
                copy=True,
                whiten=False,
                svd_solver="auto",
                tol=0.0,
                iterated_power="auto",
                random_state=rand_state,
            )
            pca.fit(data)
            # Do reduced PCA based on original fit; return dim reduced data
            target_error = 1 - n_components_or_expl_variance
            num_components = np.max(
                np.argwhere(pca.explained_variance_ratio_ > target_error)
            )
        pca = PCA(
            n_components=num_components,
            copy=True,
            whiten=True,
            svd_solver="auto",
            tol=0.0,
            iterated_power="auto",
            random_state=rand_state,
        )
        pca.fit(data)
    return pca.transform(data), pca


def get_fft_information(trials, n_components):
    """
    Extract fft of each trial, subsample to n_components
    Parameters
        trials: [list] with each element being an eyetrace object
        n_components: [int] number of FT components to subsample and add to data.
    TODO: give n_components a default
    """
    # calcualte Power Spectrums
    ft_x = [np.abs(np.fft.fft(trial.x)) ** 2 for trial in trials]
    ft_y = [np.abs(np.fft.fft(trial.y)) ** 2 for trial in trials]
    # subsample to number of desired components
    n_samples = n_components // 2
    # resample x and y fts, and concatenate them
    # NOTE: The line below generated a futures warning about indexing with tuples. Don't have time to spend
    # figuring out why, (it's soemthing in the scipy.stats pacakage) but this code works for now.
    # the warning might go away with a future update of scipy.
    ss_fts = [
        np.concatenate(
            (
                scipy.signal.resample(np.asarray(ft_x[i]), n_samples),
                scipy.signal.resample(np.asarray(ft_y[i]), n_samples),
            )
        )
        for i in range(len(ft_x))
    ]
    # rescale
    ss_fts = rescale_data_to_one(ss_fts)
    return np.asarray(ss_fts)


def get_vel_information(trials):
    """
    Extract velocity vals in 'interesting region', subsample to n_components
    Parameters
        trials: [list] with each element being an eyetrace object
    """
    # calcualte Mean Velocity per trace
    vs = np.expand_dims(
        np.asarray([np.mean(np.asarray(trial.v)) for trial in trials]), axis=-1
    )
    # rescale
    vs = rescale_data_to_one(vs)
    return vs


def split_trials(trials_list, multiplier):
    """
    Split list of trials into *multiplier* more trials, at *1/multiplier* the time length.
    In the event that *multiplier* does not divide evenly into a trial,
        then the last multiple will be shorter (i.e. the remainder)

    Parameters:
        trials_list: List of eyetrace instances
        multiplier: How many pieces to break each trial into
    Returns:
        short_trials_list: List of eyetrace instances, shorter than trials_list
    """
    if np.any([len(trial.x) % multiplier != 0 for trial in trials_list]):
        print(
            "split_trials: WARNING: Splits will not be equal length because multiplier does not divide into trial length"
        )
    short_trials_list = []
    for trial in trials_list:
        idx_len = len(trial.x) // multiplier  # integer division, flooring
        for i in range(multiplier):
            idx_start = idx_len * (i)
            idx_end = idx_len * (i + 1)
            xvals = trial.xraw[idx_start:idx_end]
            yvals = trial.yraw[idx_start:idx_end]
            tvals = trial.time[idx_start:idx_end]
            vvals = trial.v[idx_start:idx_end]
            interp_times = trial.interp_times
            blink_times = trial.blink_times
            blink_num = trial.blink_num
            ppd = trial.ppd
            filename = (
                trial.fname_full + "_subset_" + str(i + 1) + "_of_multiplier"
            )
            subinfo = trial.subinfo
            newtrial = EyeTrace(
                xvals,
                yvals,
                tvals,
                vvals,
                interp_times,
                blink_times,
                blink_num,
                ppd,
                filename,
                subinfo,
            )
            short_trials_list.append(newtrial)
    return short_trials_list
