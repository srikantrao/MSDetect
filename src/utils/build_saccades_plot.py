import numpy as np
import pandas as pd 
import streamlit as st 
import pre_processing
import matplotlib.pyplot as plt 
import plot_utils
"""
Test out the **build_dict** function on the files
"""
pat_path = "./patient_saccades_data.csv"
con_path = "./control_saccades_data.csv"

# Build the dict
subject_dict = pre_processing.build_dict(con_path, row = 9)
subject_dict = pre_processing.in_subject_mean_std(subject_dict)

# for key, value in subject_dict.items():
# st.write(key, value)

st.write(f"Number of Patient Subjects: {len(subject_dict)}")

"""
Plot the In subject mean and variance 
"""
array = np.array([[value[0], value[1]] for key, value in subject_dict.items()])
st.write(array.shape)

# Pretty print the in-sample
fig = plot_utils.set_plot_properties()
plt.plot(array[:, 0], 'bo-', label="Subject Mean")
plt.plot(array[:, 1], 'mx-', label = "Subject Sigma")
plt.legend(loc="upper right")
plt.title("Within Control Metrics")
plt.xlabel("Subject ID")
plt.ylabel("Micro-saccades")
st.pyplot()


