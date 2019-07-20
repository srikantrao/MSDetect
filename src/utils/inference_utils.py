""" Helper fulctions used in inference.py """


# Import Statements 
import numpy as np
import pandas as pd
import streamlit as st
from utils import data_manipulation
import os 
import re
import glob


def show_ui(file_list, pat_str_list):
    """
    Return the filtered file_list (should be only 1 by the end of it). 
    """
    if not file_list:
        st.warning('This Patient ID does not exist. Please reselect.')
        return None
    index = st.selectbox(label = "Select the Patient ID",
                               options = pat_str_list)

    # second select box
    ver_list = get_ver_list
    if not options_2:
        st.warning('No options for %s. Please select another.' % selection_1)
        return None
    selection_2 = st.selectbox('selection for %s' % selection_1, options_2)

    # third select box
    options_3 = get_options_for_select_box_3(selection_2)
    if not options_3:
        st.warning('No options for %s. Please select another.' % selection_2)
        return
    selection_3 = st.selectbox('selection for %s' % selection_2, options_3)
    return (selection_1, selection_2, selection_3)


def filter_by_eye(file_list, eye):
    """
    Filter the list of files by which eye trace
    @param file_list: List of file paths 
    @param version: str which is either "r" or "l" to filter the list of files
    @returns: List of file paths that match the particular eye
    """
    new_file_list = []
    for fi in file_list:
        fi_name = os.path.basename(fi)
        if fi_name.split("_")[0][-1].lower() == eye:
            new_file_list.append(fi)

    return new_file_list

def load_patient_ids(patient_file_path):
    """
    Load all the patient ids from the csv 
    @param: str which is the path to the patient information
    """
    df = pd.read_csv(patient_file_path)
    patient_list = df['Subject ID'].tolist()
    pat_str_list = ["Patient ID: " + str(pat) for pat in patient_list]

    return patient_list, pat_str_list

def get_ver_list(file_list):
    """
    Get a list of the version that are available for a particular list of files
    @param file_list: List of file paths
    @returns: List of versions that are available 
    """

    ver_list = [os.path.basename(fi).strip().split("_")[1] for fi in file_list]
    ver_list = list(set(ver_list))
    ver_list = [f"Version: {v[-1]}" for v in ver_list]

    return ver_list

def get_eye_list(file_list):
    """
    Get list of eyes that can be selected. (Right and / or Left eye)
    @param : List of file paths 
    @returns: Eye list 
    """
    eye_list = [os.path.basename(fi).strip().split("_")[0][-1] for fi in file_list]
    eye_list = list(set([eye.lower() for eye in eye_list]))
    eye_str_list = ["Right Eye" if eye == 'r' else "Left Eye" for eye in eye_list]

    return eye_list, eye_str_list


def get_file_list(subject_id):
    """
    Get the list of files which are the traces of that particular subject id 
    @param subject_id: str which represents the subject ids traces to look for 
    @returns: List of file paths 
    """
    file_list = glob.glob(os.path.join(os.environ["PATDATA"],
                                       f"*{subject_id}*"))

    return file_list

def filter_by_version(file_list, version):
    """
    Filter the list of files by version
    @param file_list: List of file paths 
    @param version: str to filter the list of files using
    @returns: list of file paths that are of that particular version
    """
    filter_string = f"V00{version[-1]}"

    file_list = list(filter(lambda s: filter_string in s, file_list))

    return file_list

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
