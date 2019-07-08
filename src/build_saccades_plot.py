import numpy as np
import pandas as pd 
import streamlit as st 
from utils import pre_processing
import matplotlib.pyplot as plt 
from utils import plot_utils
"""
Test out the **build_dict** function on the files
"""
pat_path = "./data/patient_saccades_data.csv"
con_path = "./data/control_saccades_data.csv"

# Build the dict
subject_dict = pre_processing.build_dict(pat_path, row = 3)
subject_dict = pre_processing.in_subject_mean_std(subject_dict)

# for key, value in subject_dict.items():
# st.write(key, value)

def autolabel(rects, xpos='center'):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0, 'right': 1, 'left': -1}

    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.1f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(offset[xpos] * 3, 3),  # use 3 points offset
                    textcoords="offset points",  # in both directions
                    ha=ha[xpos], va='bottom')


st.write(f"Number of Patient Subjects: {len(subject_dict)}")

"""
Plot the In subject mean and variance 
"""
array = np.array([[value[0], value[1]] for key, value in subject_dict.items()])
st.write(array.shape)

# Pretty plot the in-sample
indices = np.arange(20)
width = 0.35
fig, ax = plot_utils.set_plot_properties()
rects1 = ax.bar(indices - width, array[0 : 20, 0], width, 
                color = 'orange', yerr = array[0 : 20, 1], label = "Mean")
plt.legend(loc="upper right")
plt.title("Within Patient Metrics")
plt.xlabel("20 randomly Patients in trial")
plt.ylabel("Number of Micro-saccades")

autolabel(rects1, "left")
plt.tight_layout()
st.pyplot()
