import pytest
import numpy as np
import pandas as pd 
from utils import pre_processing

def test_dummy():
    """
    Test setup 
    """
    assert True

def test_downsample():
    """
    Test the downsample function 
    """

    # Default axis of downsample is 2. 

    X = np.random.randint(0, 100, size = (10, 2, 1000))

    X = pre_processing.downsample_traces(X, 200)

    assert X.shape == (10, 2, 200)

def test_flip():
    """
    Test the flip function 
    """
    X = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12],
                  [13, 14, 15, 16]])
    Y = np.array([1, 0, 0, 1])

    sub_id = np.array([34, 54, 24, 67])

    X_test, y_test, sub_id_test = pre_processing.flip_augment(X, Y, sub_id)
    X_ans = np.array([[-1, -2, -3, -4],
                      [-5, -6, -7, -8],
                      [-9, -10, -11, -12],
                      [-13, -14, -15, -16]])
    
    assert np.array_equal(X_test, X_ans)


def test_use_dict():
    """
    If you are planning to use only distance travelled as one of the input
    channels, use_dict will have to be used. 
    Test out to make sure function is doing what is expected. 
    """


