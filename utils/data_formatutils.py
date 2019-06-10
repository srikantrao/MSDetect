import copy
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import utils.eyetrace as eyetrace
import scipy.signal as signal

def make_twpca_eyetrace_matrix(pat_trial_list, con_trial_list):
    '''
    Take lists of patient and control eyetraces and inserts into (tracenumber, timepoint, 2)
    numpy array with nans padding any traces shorter than the max length.
    In the final dimension, 0 is for x locations and 1 is for y.
    Also returned is the last index containing patient traces. The rest are controls.
    This is the appropriate format for TWPCA

    Parameters:
        pat_trial_list: List of patient eyetraces - each list element is an instance of eyetrace class
        con_trial_list: List of control eyetraces - each list element is an instance of eyetrace class
    Returns:
        traials_array: a 3d Numpy array indexed by [tracenumber, timepoint, x_or_y] where 0 is x, 1 is y
        pat_trial_max: the last index containing patient traces. The rest are controls.
    '''
    # save index where patients start and controls begin
    pat_trial_max = len(pat_trial_list) - 1
    # make a list of both trials and controls together
    trial_eyetrace_list = pat_trial_list + con_trial_list

    # calculate the max length of a trial so we can pad with nans
    maxlen = np.max([np.max((len(trial.x), len(trial.y))) for trial in trial_eyetrace_list])
    # arrray is indexed by [tracenumber, timepoint, x_or_y] where 0 is x, 1 is y
    trials_array = np.zeros((len(trial_eyetrace_list), maxlen, 2))
    trials_array[...] = np.nan

    #fill in our array
    for i, trial in enumerate(trial_eyetrace_list):
        trials_array[i, :len(trial.x), 0] = trial.x
        trials_array[i, :len(trial.y), 1] = trial.y

    return(trials_array, pat_trial_max)

def make_pca_eyetrace_matrix(pat_trial_list, con_trial_list, interleave=True):
    '''
    Take lists of patient and control eyetraces and inserts into (tracenumber, timepoint*2)
    numpy array with any traces longer than the min length cutttoff.
    Also returned is a list marking patient trials (1), and controls trials (0).

    Parameters:
        pat_trial_list: List of patient eyetraces - each list element is an instance of eyetrace class
        con_trial_list: List of control eyetraces - each list element is an instance of eyetrace class
    Returns:
        traials_array: a 3d Numpy array indexed by [tracenumber, timepoint]
        pat_trial_flag: flag patients vs controls.
    '''
    # flag for patients & controls
    pat_trial_flag = [1]*len(pat_trial_list) + [0]*len(con_trial_list)
    # make a list of both trials and controls together
    trial_eyetrace_list = pat_trial_list + con_trial_list

    # calculate the min length of a trial so we can throwout any that are greater
    minlen = np.min([np.min((len(trial.x), len(trial.y))) for trial in trial_eyetrace_list])
    # arrray is indexed by [tracenumber, timepoint*2], where *2 comes from x/y
    trials_array = np.zeros((len(trial_eyetrace_list), minlen*2))

    # fill in our array
    # if we are interleaving x and y, do so
    if(interleave):
        for i, trial in enumerate(trial_eyetrace_list):
            trials_array[i, ::2] = trial.x[:minlen]
            trials_array[i, 1::2] = trial.y[:minlen]
    # otherwise just append the two
    else:
        for i, trial in enumerate(trial_eyetrace_list):
            trials_array[i,:] = np.concatenate((trial.x[:minlen], trial.y[:minlen]))

    return(trials_array, pat_trial_flag)

def flip_data_on_y_axis(data):
    """
    Flips data on the specified axis
    Parameters:
        data: np.ndarray of shape [num_datapoints, num_timepoints, dim] where dim is (x,y)
    Returns:
        flipped_data: np.ndarray of the same shape as the input,
        with the dim specified by flip_axis flipped
    """
    num_datapoints, num_timepoints, num_dim = data.shape
    flipped_data = np.zeros_like(data)
    for datapoint in range(num_datapoints):
        for timepoint in range(num_timepoints):
            data_matrix = data[datapoint, timepoint, :]
            y_reflect = np.asarray([[-1,0],[0,1]])
            data_refl_matrix = np.dot(y_reflect, data_matrix)
            flipped_data[datapoint, timepoint, :] = data_refl_matrix
    return flipped_data

def flip_trials_y(trials):
    """
    Flips trials on the y axis
    NOTE: This function should only be applied to training data
    Parameters:
        trials: eyetrace object
    Returns:
        flipped_trials: eyetrace object with the y axis flipped
    """
    flipped_trials = copy.deepcopy(trials)
    for trial in flipped_trials:
        y_reflect = np.asarray([[-1,0],[0,1]])
        xy_matrix = np.stack([trial.x, trial.y], axis=0)
        xy_refl_matrix = np.dot(y_reflect, xy_matrix)
        trial.x = xy_refl_matrix[0]
        trial.y = xy_refl_matrix[1]
    return flipped_trials

def one_hot_to_dense(one_hot_labels):
    """
    Convert list of dense labels to a matrix of one-hot labels
    Parameters:
        one_hot_labels: one-hot numpy array of shape [num_labels, num_classes]
    Returns:
        dense_labels: 1D numpy array of labels
            The integer value indicates the class and 0 is assumed to be a class.
            The integer class also indicates the index for the corresponding one-hot representation
    """
    one_hot_labels = np.asarray(one_hot_labels)
    num_labels, num_classes = one_hot_labels.shape
    dense_labels = np.zeros(num_labels)
    dense_labels = np.squeeze(np.asarray([np.argwhere(one_hot_labels[label_id,:]==1).item()
      for label_id in range(num_labels)]))
    return dense_labels

def dense_to_one_hot(dense_labels):
    """
    Convert list of dense labels to a matrix of one-hot labels
    Parameters:
        dense_labels: list or 1D numpy array of labels
            The integer value indicates the class and 0 is assumed to be a class.
            The integer class also indicates the index for the corresponding one-hot representation
    Returns:
        one_hot_labels: one-hot numpy array of shape [num_labels, num_classes]
    """
    dense_labels = np.asarray(dense_labels)
    num_labels = int(dense_labels.size)
    num_classes = int(np.amax(dense_labels) + 1) # 0 is a class
    one_hot_labels = np.zeros((num_labels, num_classes))
    index_offset = np.arange(num_labels, dtype=np.int32) * num_classes
    one_hot_labels.flat[index_offset + dense_labels.ravel()] = 1
    return one_hot_labels

def make_mlp_eyetrace_matrix(pat_trial_list, con_trial_list, fft_subsample):
    '''
    Take lists of patient and control eyetraces and inserts into (tracenumber, timepoint, dim)
    where dim is (x,y)
    numpy array with any traces longer than the min length cutttoff.
    Also returned is a list marking patient trials (1), and controls trials (0).

    Parameters:
        pat_trial_list: List of patient eyetraces - each list element is an instance of eyetrace class
        con_trial_list: List of control eyetraces - each list element is an instance of eyetrace class
        fft_subsample: [int] number of FT components to subsample and add to data.
    Returns:
        traials_array: a 3d Numpy array indexed by [tracenumber, timepoint, dim] where dim=2 indicates [y,x]
        pat_trial_flag: flag patients vs controls.
        stats: dictionary of relevant statistics for each datapoint
    '''
    # save index where patients start and controls begin
    pat_trial_flag = np.asarray([1,]*len(pat_trial_list)+[0,]*len(con_trial_list))
    # make a list of both trials and controls together
    trial_eyetrace_list = pat_trial_list + con_trial_list
    stats = {
        "ages":get_age_information(trial_eyetrace_list),
        "ffts":get_fft_information(trial_eyetrace_list, fft_subsample),
        "velocities":get_vel_information(trial_eyetrace_list),
        "nblinks":get_nblinks_information(trial_eyetrace_list)}
    # calculate the min length of a trial so we can pad with nans
    minlen = np.min([np.min((len(trial.x), len(trial.y))) for trial in trial_eyetrace_list])
    # arrray is indexed by [tracenumber, timepoint, dim] where dim=2 is (x,y)
    trials_array = np.zeros((len(trial_eyetrace_list), minlen, 2))
    # fill in our array
    for i, trial in enumerate(trial_eyetrace_list):
        trials_array[i,:,0] = trial.x[:minlen]
        trials_array[i,:,1] = trial.y[:minlen]
    return(trials_array, pat_trial_flag, stats)

def format_mlp_data(data, labels, stats, params):
    """
    Assemble input data into a list of train, validation, and test sets
    Input labels are converted to 1-hot labels and also placed in a list accordingly
    Parameters:
        data: np.ndarray of shape [num_datapoints, ...]
        labels: np.ndarray of shape [num_datapoints] (where the int value indicates class)
        stats: dictionary containing keys {"ages", "ffts", "velocities", "nblinks"}
        params: class containing the following member variables:
          num_train: how many datapoints to use for training
          num_val: how many datapoints to use for validation
          num_test: how many datapoints to use for testing
          rand_state: np.random.RandomState()
    Returns:
        data_list: list of np.ndarray with training first, then validation, then test
        one_hot_labels: a list of one-hot np.ndarray of shape [num_datapoints, num_classes]
            with training first, then validation, then test
        stats: list of dictionaries with datapoint statistics
    """
    num_train = params.num_train
    num_val = params.num_val
    num_test = params.num_test
    rand_state = params.rand_state
    labels = np.asarray(labels)
    labels1h = dense_to_one_hot(labels)
    num_datapoints = data.shape[0]
    num_labels = labels1h.shape[0]
    assert num_datapoints == num_labels, (
        "The number of labels must match the number of datapoints")
    if num_train is None:
        num_train = num_labels - num_val - num_test
    assert num_labels >= num_train+num_val+num_test, (
        "The total number of labels must be greater than or equal to num_train+num_val+num_test")
    all_indices = np.arange(num_labels)
    num_classes = np.amax(labels) + 1
    num_test_per_class = int(num_test / num_classes)
    num_val_per_class = int(num_val / num_classes)
    label_counts = []
    test_indices_list = []
    val_indices_list = []
    for label in range(num_classes):
        label_indices = np.argwhere(labels==label).ravel()
        label_counts.append(label_indices.size)
        # Check to see if we need to repeat sample draws to have enough datapoints
        if num_test_per_class <= label_counts[label]:
            test_replace=False
            if num_test_per_class + num_val_per_class <= label_counts[label]:
                val_replace=False
            else:
                val_replace=True
                print("WARNING: format_mlp_data: not enough labeled examples to populate num_test and num_val.")
        else:
            test_replace=True
            print("WARNING: format_mlp_data: not enough labeled examples to populate num_test.")
        if num_test_per_class > 0:
          # Randomly draw samples from entire dataset for test set
          test_indices_list.append(rand_state.choice(label_indices, num_test_per_class,
              replace=test_replace))
        if num_val_per_class > 0:
          # Randomly draw samples from dataset minus test samples for val set
          if num_test_per_class > 0:
              val_draw_set = [idx for idx in label_indices if idx not in test_indices_list[label]]
          else:
              val_draw_set = [idx for idx in label_indices]
          val_indices_list.append(rand_state.choice(val_draw_set, num_val_per_class,
              replace=val_replace))
    if num_test_per_class > 0:
        test_indices = np.concatenate(test_indices_list)
    else:
        test_indices = []
    if num_val_per_class > 0:
        val_indices = np.concatenate(val_indices_list)
    else:
        val_indices = []
    train_indices = rand_state.choice([idx
        for idx in all_indices
        if idx not in np.union1d(test_indices, val_indices)], num_train, replace=False)
    out_data = [data[train_indices, ...], data[val_indices, ...], data[test_indices, ...]]
    out_labels = [labels1h[train_indices,...], labels1h[val_indices,...], labels1h[test_indices,...]]
    out_stats = [{key:value[train_indices] for (key, value) in stats.items()},
        {key:value[val_indices] for (key, value) in stats.items()},
        {key:value[test_indices] for (key, value) in stats.items()}]
    return (out_data, out_labels, out_stats)

def make_tsfresh_dataframes(pat_trial_list, con_trial_list):
    '''
    Take lists of patient and control eyetraces and convert to a pandas data frame (long format)
    compatable with ts-fresh. Format is with columns: trial_id, time, x_pos, y_pos.

    Parameters:
        pat_trial_list: List of patient eyetraces - each list element is an instance of eyetrace class
        con_trial_list: List of control eyetraces - each list element is an instance of eyetrace class
    Returns:
        traials_df: a 3d Numpy array indexed by [tracenumber, timepoint]
        pat_trial_flag: flag patients vs controls.

    '''

    # make a list of both trials and controls together
    trial_eyetrace_list = pat_trial_list + con_trial_list
    # calculate the max length of a trial so we can pad with nans
    minlen = np.min([np.min((len(trial.x),len(trial.y))) for trial in trial_eyetrace_list])
    # create a dataframe to hold time seris & one to flag patient status
    trials_df = pd.DataFrame()
    trials_patflag = pd.DataFrame()
    # fill out dataframes out - first patients
    i=np.float(1)
    for trial in pat_trial_list:
        trial_data = pd.DataFrame(np.vstack((np.repeat(i,len(trial.time)),
                                             trial.time.astype('float'),
                                             trial.x.astype('float'),
                                             trial.y.astype('float'))).T)
        trials_df = trials_df.append(trial_data, ignore_index=True)
        patflag_data = pd.DataFrame(np.vstack((i,1)).T,
                                    columns=('id','patient_flag'))
        trials_patflag = trials_patflag.append(patflag_data, ignore_index=True)
        i+=1
    # then controls
    for trial in con_trial_list:
        trial_data = pd.DataFrame(np.vstack((np.repeat(i,len(trial.time)),
                                             trial.time.astype('float'),
                                             trial.x.astype('float'),
                                             trial.y.astype('float'))).T)
        trials_df = trials_df.append(trial_data, ignore_index=True)
        patflag_data = pd.DataFrame(np.vstack((i,0)).T,
                                    columns=('id','patient_flag'))
        trials_patflag = trials_patflag.append(patflag_data, ignore_index=True)
        i+=1
    # name our coumns
    trials_df.columns=['id','time','x','y']

    return(trials_df, trials_patflag)

def truncate_trials(trial_list):
    '''
    Truncates the trials to be the length of the shortest trial

    Parameters:
        trials_list: List of eyetrace instances
    Returns:
        truncated_trial_list: List of eyetrace instances, with the time data truncated to the shortest instance
    '''
    min_length = np.min([len(trial.x) for trial in trial_list])
    truncated_trial_list = []
    for trial in trial_list:
        xvals = trial.xraw[:min_length]
        yvals = trial.yraw[:min_length]
        tvals = trial.time[:min_length]
        vvals = trial.v[:min_length]
        interp_times = trial.interp_times
        blink_times = trial.blink_times
        blink_num = trial.blink_num
        ppd = trial.ppd
        filename = trial.fname_full+'_truncated_'+str(min_length)
        subinfo = trial.subinfo
        newtrial = eyetrace.EyeTrace(xvals, yvals, tvals, vvals, interp_times, blink_times, blink_num, ppd, filename, subinfo)
        truncated_trial_list.append(newtrial)
    return truncated_trial_list

def split_trials(trials_list, multiplier):
    '''
    Split list of trials into *multiplier* more trials, at *1/multiplier* the time length.
    In the event that *multiplier* does not divide evenly into a trial,
        then the last multiple will be shorter (i.e. the remainder)

    Parameters:
        trials_list: List of eyetrace instances
        multiplier: How many pieces to break each trial into
    Returns:
        short_trials_list: List of eyetrace instances, shorter than trials_list
    '''
    if np.any([len(trial.x) % multiplier != 0 for trial in trials_list]):
        print("split_trials: WARNING: Splits will not be equal length because multiplier does not divide into trial length")
    short_trials_list = []
    for trial in trials_list:
        idx_len = len(trial.x) // multiplier # integer division, flooring
        for i in range(multiplier):
            idx_start = idx_len*(i)
            idx_end = idx_len*(i+1)
            xvals = trial.xraw[idx_start:idx_end]
            yvals = trial.yraw[idx_start:idx_end]
            tvals = trial.time[idx_start:idx_end]
            vvals = trial.v[idx_start:idx_end]
            interp_times = trial.interp_times
            blink_times = trial.blink_times
            blink_num = trial.blink_num
            ppd = trial.ppd
            filename = trial.fname_full+'_subset_'+str(i+1)+'_of_multiplier'
            subinfo = trial.subinfo
            newtrial = eyetrace.EyeTrace(xvals, yvals, tvals, vvals, interp_times, blink_times, blink_num, ppd, filename, subinfo)
            short_trials_list.append(newtrial)
    return (short_trials_list)
def split_training_test(trials_list, percent_holdout=10, rand_state=np.random.RandomState(None)):
    '''
    Split a list of trials into a randomly chosen training and test set
    with percent_holdout percent of trials heldout for testing

    Parameters:
        trials_list: list of eyetrace instances
        percent_holdout: number in percentage of data to keep as test
    Returns:
        train_trials: list of eyetrace instances leftover for training
        test_trials: list of eyetrace instances for testing
    '''
    n_holdout = len(trials_list)* percent_holdout//100
    shuffle = np.arange(len(trials_list))
    rand_state.shuffle(shuffle)
    test_trials = [trials_list[i] for i in shuffle[:n_holdout]]
    train_trials = [trials_list[i] for i in shuffle[n_holdout:]]
    return (train_trials, test_trials)

def rescale_data_to_one(data):
    '''
    Rescales input data to be between 0 and 1
    TODO: Verify that this is actually the right thing to do?
        We want all of the data features to be on the same scale
        Should this normalization be done per sample or across the whole dataset?
    '''
    data_min = np.amin(data)
    data_max = np.amax(data)
    return (data - data_min) / (data_max - data_min)

def pca_whiten_reduce(data, n_components_or_expl_variance=0.995, rand_state=np.random.RandomState(None), pca=None):
    '''
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
    '''
    # Reshape data to 2D if it is more than 2D
    if data.ndim > 2:
        data = data.reshape(data.shape[0], np.prod(data.shape[1:]))
    if pca is None:
        if n_components_or_expl_variance >= 1.0:
            num_components = int(n_components_or_expl_variance)
        else:
          # Do full dim PCA to figure out how many components to keep
          pca = PCA(n_components=None, copy=True, whiten=False, svd_solver="auto",
              tol=0.0, iterated_power='auto', random_state=rand_state)
          pca.fit(data)
          # Do reduced PCA based on original fit; return dim reduced data
          target_error = 1 - n_components_or_expl_variance
          num_components = np.max(np.argwhere(pca.explained_variance_ratio_ > target_error))
        pca = PCA(n_components=num_components, copy=True, whiten=True, svd_solver="auto",
            tol=0.0, iterated_power="auto", random_state=rand_state)
    return pca.fit_transform(data), pca

def get_age_information(trials):
    '''
    Extract age, convert to a float between 0 and 1
    Parameters
        trials: [list] with each element being an eyetrace object
    '''
    ages = [int(trial.sub_age.item()) for trial in trials]
    max_age =  np.max(ages)
    min_age =  np.min(ages)
    ages = np.asarray((ages - min_age) / (max_age - min_age + 1e-10))[:,None]
    ages = rescale_data_to_one(ages)
    return ages

def get_fft_information(trials, n_components):
    '''
    Extract fft of each trial, subsample to n_components
    Parameters
        trials: [list] with each element being an eyetrace object
        n_components: [int] number of FT components to subsample and add to data.
    TODO: give n_components a default
    '''
    # calcualte Power Spectrums
    ft_x = [np.abs(np.fft.fft(trial.x))**2 for trial in trials]
    ft_y = [np.abs(np.fft.fft(trial.y))**2 for trial in trials]
    # subsample to number of desired components
    n_samples = n_components//2
    # resample x and y fts, and concatenate them
    ## NOTE: The line below generated a futures warning about indexing with tuples. Don't have time to spend
    ## figuring out why, (it's soemthing in the scipy.stats pacakage) but this code works for now.
    ## the warning might go away with a future update of scipy.
    ss_fts = [np.concatenate((signal.resample(np.asarray(ft_x[i]), n_samples),
        signal.resample(np.asarray(ft_y[i]), n_samples))) for i in range(len(ft_x))]
    # rescale
    ss_fts = rescale_data_to_one(ss_fts)
    return np.asarray(ss_fts)

def get_vel_information(trials):
    '''
    Extract velocity vals in 'interesting region', subsample to n_components
    Parameters
        trials: [list] with each element being an eyetrace object
    '''
    # calcualte Mean Velocity per trace
    vs = np.expand_dims(np.asarray([np.mean(np.asarray(trial.v)) for trial in trials]),axis=-1)
    # rescale
    vs = rescale_data_to_one(vs)
    return vs

def get_nblinks_information(trials):
    '''
    Extract number of blinks
    Parameters
        trials: [list] with each element being an eyetrace object
    '''
    # extract number of blinks per trace
    blinks = [trial.blink_num for trial in trials]
    # rescale
    #print(blinks, np.shape(blinks))
    blinks = rescale_data_to_one(blinks)
    return blinks
