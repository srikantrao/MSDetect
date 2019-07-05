## MSDetect - Eye tracking for detection of Mutliple Sclerosis

This repo contains the work done as an Insight Artifical Intelligence Fellow, consulting for a company that is using state of the art high-resolution pupil tracking hardware combined with AI to change the way Multiple Sclerosis is diagnosed and tracked in patients. 

Recent medical literature has explored the relationship between macular degeneration and the progression of multiple sclerosis in patients[1].

The current model uses a model that uses engineering features that are captured by hand-labelling pupil trace data. 

I worked on exploring if high-resolution vertical and horizontal motion of the eye can be used to detect the presence of, as well as track the progression of Multiple Sceloris in patients.

Since this work was done as part of a clinical trial, the data, the final models used as well as a subset of results achieved cannot be shared. 
 
###  Environment Setup  

#### Ubuntu 18.04 

1. Download and nstall Anaconda for python 3.7

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Run the following command from the terminal 

```
conda env create -f environment_linux.yml
conda activate msdetect
```

#### Running on Mac 

```
conda env create -f environment_mac.yml
conda activate msdetect

```
### Running the Code

#### Training

Training on one of the models can be performed by using the following commands. 

```commandline
python src/train.py ms-classfication 
```

A summary of the additional parameters is provided below - 

##### Training parameters 
 - `--data_dir`
     - Path to the directory which contains *.mat or *.npy pupil traces.
 - `--patient_file_path`
     - Path to patient health information file. 
 - `--model_path`
    - Path to which the model should be saved. By default it gets saved in the `params` folder. 
- `model_name`
    - Model name of the saved model.  
 - `--num_layers`
    - Number of layers to use in the ResNet model. Default value is 8.
 - `--lr`
    - Learning rate to be used for training. Default value is 0.0005 
 - `--drop_rate`
    - Dropout rate to be used for the Conv layers. Default value is 0.2 
 - `--num-folds`
   - Number of folds to use for cross validation. Default value is 1. 
 - `--num-splits`
   - Number of splits to break up each trace into. Default value is 1. Changing this parameter is not recommended.
 - `--random_seed`
   - Random Seed to be used. 
 - `--test_fraction`
	- Fraction of samples to be used for testing.
 - `--filter-age`
   - If included filter out any subjects whose age is >= the filter age
 - `--batch_size`
    Batch Size to be used for training. Default value is 32.
 - `--early_stopping`
    Switch to enable or disable   
- `--flip-y`
   - Whether or not to duplicate the trace data by flipping the traces.
 - `--test-fraction`
   - Test set fraction. 


#### Inference  

#### Counting Saccades

### Models

Two kind of models are available can be used to determine  

### Results 

#### Micro-Saccade distribution

#### Age distribution in Controls and in Patients 

The controls are included
### Acknowledgements


### References  
[1] Sheehy et al: J Neuro-Ophthalmol 2018; 38: 488-493



