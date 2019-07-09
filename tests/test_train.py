import pytest
import numpy as np
import pandas as pd 
import os
from utils import pre_processing 
from utils import data_manipulation
from utils import validation 
from keras.models import load_model
from models.lstm_model import LSTM_Model
from keras.models import Model

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


def test_saved_lstm_model():
    """
    Make sure that the LSTM model gets saved correctly
    """
    test_model_path = os.environ["MODELPATH"] 
    test_model_name = os.environ["TEST_MODEL"]
    test_model_name = os.path.join(test_model_path,
                                   test_model_name)
    # Load the saved model 
    test_model = load_model(test_model_name)

    assert isinstance(test_model, Model)
