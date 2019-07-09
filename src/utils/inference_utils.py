""" Helper fulctions used in inference.py """


# Import Statements 
from utils import data_manipulation
import os 
import re


def read_single_trace(filepath, patient_file_path, start_index = 10):
    """
    Generate an EyeTrace object by procesing the file located 
    at filepath.
    returns: Eyetrace object 
    """

    # Read in the Patient information
    pf = data_manipulation.read_pstats(patient_file_path)

    # Gather the subject Info - Need this for the EDSS values 
    pattern = re.compile(r'\d+')
    fname = os.path.split(filepath)[-1]
    match = re.search(pattern, fname)
    if not match:
        print(f"Warning: Could not parse subject id for file {fname}")

    subj_id = match[0]
    subinfo = pf[pf["Subject ID"] == subj_id]
    
    # Read in the trace
    trace = data_manipulation.read_n_filter(filepath, subinfo, 
                                            version = 9, start_index = 10)

    return trace
