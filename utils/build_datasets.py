import utils.data_formatutils as dfu
from utils.downsample import DatasetAtFrequency
import os
import utils.readinutils as readinutils
import sys
sys.path.append('../')

# Params
from params.model_params import params

"""--------------------------------------
  Data path specification is done here:
--------------------------------------"""
datadir = "/data/envision_working_traces"
patientfile = './data/patient_stats.csv'
"""--------------------------------------
--------------------------------------"""


def make_datasets(freqs):
    """
    Creates DatasetGroup at listed frame-rates (frequencies).
       freqs: list
    Returns:
       Dictionary of downsamples datasets, indexed by frequency, with values
       in the form:
              [patient_trails, control_trials]
    """

    # Data setup
    trials = readinutils.readin_traces(datadir, patientfile)
    trials = [trial for trial in trials if(trial.sub_ms.size > 0)]
    if params.truncate_trials:
        trials = dfu.truncate_trials(trials)
    if params.trial_split_multiplier is not None:
        trials = dfu.split_trials(trials,
                                  multiplier=params.trial_split_multiplier)

    patient_trials = [trial for trial in trials if trial.sub_ms == '1']
    control_trials = [trial for trial in trials if trial.sub_ms == '0']

    new_datasets = {}
    full_dataset = [patient_trials, control_trials]

    # Build datasets at specified frequencies by downsampling eye-traces
    print("Building datasets...")
    for freq in freqs:
        data = DatasetAtFrequency(freq, full_dataset)
        new_datasets[freq] = data.get_new_data()

    return new_datasets
