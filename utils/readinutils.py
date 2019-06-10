from glob import glob
import os
from scipy.io import loadmat
import numpy as np
import pandas as pd
import utils.eyetrace as eyetrace

def read_n_filter(filename, subinfo):
    '''
    Read in our files, filter them, and return an eyetrace instance
    Parameters:
        Filename (str) Location of the .mat file containig the eyetrace
        subinfo (PD array)    line from pd datasheet containing subject information
    Returns:
        trial (Eyetrace) Eyetrace instance for a given trial (variable len)
    '''
    ppd = 102 #pixels per degree
    #ppc_arcmin = 1. / ppd * 60. # Dont use conversion from pixels to arcmins
    # load file
    nfile = loadmat(filename)

    # extract features
    xss = nfile['xmotion'][0]
    yss = nfile['ymotion'][0]
    tss = nfile['timesecs'][0]
    vss = nfile['velocity'][0]
    
    # blink info
    bts = nfile['blinktimes']
    bns = nfile['blinknum'][0]

    # identify bad timepoints but do NOT delete them, they are being interpolated before readin.
    interp_times = nfile['badtimes'][0]

    # convert file contents into EyeTrace object
    trial = eyetrace.EyeTrace(xss, yss, tss, vss, interp_times, bts, bns, ppd, filename, subinfo)
    return(trial)

def read_pstats(fpath):
    '''
    Read in the Patient Info Datasheet
    Parameters:
        fpath (str) Path to patient info sheet
    Returns:
        pf (pandas df) Formatted Patient Info Datahsheet
    '''

    #read in using pandas
    pf = pd.read_csv(fpath, sep=',', dtype=object)

    #remove NAN rows
    pf = pf.dropna(axis=0, how='all', subset={'Subject ID'})

    #keep interesting columns
    keepcols = ['Subject ID','MS','Age','EDSS',
                'IS','RRMS','PP','progressive','Diseasedur',
                'Brainstem','Pyramidal','Cerebellar','Sensory']
    return(pf)

def readin_traces(dataDir, patientFilePath):
    '''
    read in set of traces
    Parameters:
        Datadir (str) Folder containing .mat files that each have eyetraces.
        patientFilePath (str) filename containing patient information
    Returns:
        Trials (list) List of trials, each an eyetrace isntance
    '''
    files = [y for x in os.walk(dataDir) for y in glob(os.path.join(x[0], '*.mat'))]

    #patient file with info
    pf = read_pstats(patientFilePath)

    #loop through and read in all files, each as a trial.
    trials = []
    for fi in files:
        #get subject info variables from datasheet
        fname = os.path.split(fi)[-1]
        subjid = fname.split("_")[0][:-1]
        subinfo = pf[pf['Subject ID'] == subjid]
        #read in our trial trace
        tial = read_n_filter(fi, subinfo)
        trials.append(tial)
    return(trials)
