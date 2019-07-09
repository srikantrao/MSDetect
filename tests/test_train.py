import pytest
import numpy as np
import pandas as pd 
import os
from utils import pre_processing 
from utils import data_manipulation
from utils import validation 

def test_readin_traces():
    """
    Make sure all of the traces are being read in. 
    """
    data_dir = os.environ['PATDATA']
    patient_file_path = os.environ['PATSTAT']
    traces = data_manipulation.readin_traces(data_dir, patient_file_path,
                                             start_index = 0)

    assert len(traces) == 794
    assert isinstance(traces[0], data_manipulation.EyeTrace)
