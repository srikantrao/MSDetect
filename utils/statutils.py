import numpy as np

import utils.data_formatutils as dfu
import tsfresh as tsfresh
import os.path
import pickle

def get_mean_velocity_vec(trial, v_dframes=1, drift_only=False, sac_only=False):
    '''
    Get the mean drift velocity vector
    
    Parameters:
        trial: eyetrace class
        v_drames: number of frames over which to compute veocity. default 1 (instentaneous veocity)
    
    Returns:
        x_vvec: vector of individual veocities
        y_vvec: vector of individual veocities

    '''
    
    x_vvec = (trial.x[v_dframes:] - trial.x[:-v_dframes])/(v_dframes*trial.dt)
    y_vvec = (trial.y[v_dframes:] - trial.y[:-v_dframes])/(v_dframes*trial.dt)
    
    x_vvec = np.append(x_vvec, x_vvec[-1])    
    y_vvec = np.append(y_vvec, y_vvec[-1])

    print(trial.sac_idx)
    if(drift_only):
        #make a new index such that we not only throw out saccades, but also any velocity data which has a saccade data point in the middle of it.
        new_sadidx = trial.sac_idx
        
        x_vvec = x_vvec[~trial.sac_idx]
        y_vvec = y_vvec[~trial.sac_idx]
    if(sac_only):
        x_vvec = x_vvec[trial.sac_idx]
        y_vvec = y_vvec[trial.sac_idx]
        
    #print(len(x_vvec))
    x_vvec = np.mean(x_vvec)
    y_vvec = np.mean(y_vvec)

    return(x_vvec, y_vvec)

def calc_mean_dtravel_x(trial, dtravel):
    xtravel = trial.x[dtravel:] - trial.x[:-dtravel]
    travel = np.mean(np.abs(xtravel))
    return(travel)

def calc_mean_dtravel_y(trial, dtravel):
    ytravel = trial.y[dtravel:] - trial.y[:-dtravel]
    travel = np.mean(np.abs(ytravel))
    return(travel)

def calc_med_dtravel(trial, dtravel):
    xtravel = trial.x[dtravel:] - trial.x[:-dtravel]
    ytravel = trial.y[dtravel:] - trial.y[:-dtravel]
    travel = np.mean(np.sqrt(np.square(xtravel) + np.square(ytravel)))
    return(travel)

def calc_med_totaltravel(trial):
    xtravel = trial.x[1:] - trial.x[:-1]
    ytravel = trial.y[1:] - trial.y[:-1]
    travel = np.median(np.sqrt(np.square(xtravel) + np.square(ytravel)))
    return(travel)

def calc_totaltravel(trial):
    xtravel = trial.x[1:] - trial.x[:-1]
    ytravel = trial.y[1:] - trial.y[:-1]
    travel = np.mean(np.sqrt(np.square(xtravel) + np.square(ytravel)))
    return(travel)

def run_tsfresh(pat_trials, con_trials, filename='features'):

    features_fpath = 'data/'+filename+'.pkl'
    filtered_features_fpath = 'data/filtered_'+filename+'.pkl'
    #check if we have extracted features, if not make them (this takes about an hour)
    if not(os.path.isfile(features_fpath)):
        
        #split into training and test, so we have data leftover to test our model on.
        patient_train, patient_test = dfu.split_training_test(pat_trials, 10)
        control_train, control_test = dfu.split_training_test(con_trials, 10)
        
        #run tsfresh on training data
        tsf_df, tsf_fdf = dfu.make_tsfresh_dataframes(patient_train, control_train)
        patient_flag_train = np.array(tsf_fdf)[:,1]
        extracted_features_train = tsfresh.extract_features(tsf_df, column_id="id", column_sort="time")
        filtered_features_train = tsfresh.select_features(extracted_features_train, patient_flag_train)
        
        #also calculate features for control data
        tsf_df, tsf_fdf = dfu.make_tsfresh_dataframes(patient_test, control_test)
        patient_flag_test = np.array(tsf_fdf)[:,1]
        extracted_features_test = tsfresh.extract_features(tsf_df, column_id="id", column_sort="time")
        #filtered_features_test = tsfresh.select_features(extracted_features_test, np.array(patient_flag_test)[:,1])
        
        #save them so we don't have to do this again!
        with open(features_fpath, 'wb') as f:
            pickle.dump([patient_train, patient_test, control_train, control_test,
             extracted_features_train, filtered_features_train, 
             patient_flag_train, extracted_features_test, patient_flag_test], f)
    #if we have the extracted features, get them
    else:
        with open(features_fpath, 'rb') as f:
            [patient_train, patient_test, control_train, control_test,
             extracted_features_train, filtered_features_train, 
             patient_flag_train, extracted_features_test, patient_flag_test] = pickle.load(f)
                    
    return(patient_train, patient_test, control_train, control_test,
           extracted_features_train, filtered_features_train, 
           patient_flag_train, extracted_features_test, patient_flag_test)

def percentage_drift(trial):
    '''
    Caclulate the percentage of time trial is drift
    Parameters: trial datastructure
    Returns: float in range [0,1] reporting fraction of time spent in drift motion
    '''

    total = len(trial.sac_idx)
    sactotal = sum(trial.sac_idx * 1)
    pdrift = (total-sactotal)/total
    
    return(pdrift)