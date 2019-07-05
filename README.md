## MSDetect - Eye tracking for detection of Mutliple Sclerosis

This repo contains the work done as an Insight Artifical Intelligence Fellow, consulting for a company that is using state of the art high-resolution pupil tracking hardware combined with AI to change the way Multiple Sclerosis is diagnosed and tracked in patients. 

Recent medical literature has explored the relationship between macular degeneration and the progression of multiple sclerosis in patients[1].
The current model uses a model that uses engineering features that are captured by hand-labelling pupil trace data. 

I worked on exploring if high-resolution vertical and horizontal motion of the eye can be used to detect the presence of, as well as track the progression of Multiple Sceloris in patients.

Since this work was done as part of a clinical trial, the data, the final models used as well as a subset of results achieved cannot be shared. 
 
###  Environment Setup  


#### Ubuntu 18.04 

1. Install Anaconda for python 3.7

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```
This downloads the Miniconda bash installer as well as 

Run the following command from the terminal 

```
conda env create -f environment_linux.yml
```
This will create a new environment called 'msdetect` with all of the necessarypackages installed. 

#### Running on Mac 

### Running the Code 


### Models 

#### Counting Saccades 


#### 1D ResNet based model 


#### LSTM based model 

### Results 

#### Micro-Saccade distribution

#### Age distribution in Controls and in Patients 

The controls are included
### Acknowledgements


### References  
[1] Sheehy et al: J Neuro-Ophthalmol 2018; 38: 488-493



