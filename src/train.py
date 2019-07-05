import sys
import logging
import os
from pathlib import Path
import fire 
import uuid
import datetime
from keras.wrappers.scikit_learn import KerasClassifier
from mord import LogisticIT
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    GroupKFold,
    GridSearchCV,
    ParameterGrid,
    GroupShuffleSplit,
)
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from xgboost import XGBClassifier
import utils.pre_processing
import utils.data_manipulation
import utils.transforms

from models.resnet_model import resnet_build_fn_closure
from utils.validation import (
    grouped_train_test_split,
    manual_grid_search_with_validation,
    grouped_accuracy,
    modeled_grouped_accuracy,
    manual_grid_search_with_validation_sklearn,
    ordinal_grid_search,
    grouped_metrics,
)

from models import lstm_model

# Set numpy print options 
np.set_printoptions(2)

# Show only Error logs for Tensorflow 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Probably used to work around this Error - This means that multiple copies of the OpenMP runtime have been linked into the program
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


logger = logging.getLogger(__name__)

class Train: 
    """
    Class to perform training on the given dataset.
    """

    def ms_classification(
        self,
        data_dir = "/data/envision_working_traces/",
        patient_file_path = "../data/patient_stats.csv",
        model_path = "../params",
        model_name = "model_ver6",
        num_layers = 8,
        lr = 0.0005,
        drop_rate = 0.2,
        num_folds = 1,
        num_splits = 1,
        flip_y = False,
        random_seed = None,
        test_fraction = 0.1,
        filter_age = None,
        batch_size = 32,
        early_stopping = True,
        epochs = 100,
        log_dir = "./logs/fit/",
        use_spectrogram = False,
        ms_only = False,
        use_lstm = False,
        verbose = True
    ):
        """
        Train a keras model (specifically, tsc.models.Resnet) to predict
        patient/control. This function will perform grouped cross
        validation over a range of hyperparameters.

        The `param_grid` which contains the hyperparameters to search is
        hardcoded.

        The full cross validation results will be saved in ~/cv_runs/ to a
        file called cv_run_JOB_ID.csv where JOB_ID is the unique ID assigned
        to the run of this function (this is the first statement that is
        logged).

        Parameters
        ----------
        data_dir
            Directory that contains the subject traces
        patient_file_path
            File that contains information about each subject
        num_folds
            Number of cross validation folds for k-fold cross validation
        num_splits
            Number of splits to break up each trace into
        flip_y
            Whether or not to duplicate the trace data by flipping the y trace
        random_seed
            Random seed to use in splitting data
        test_fraction
            Fraction of data to reserve for the test set
        filter_age
            If included filter out any subjects whose age is >= the filter_age

        """
        job_id = str(uuid.uuid4())
        logger.info(f"Job ID for this run: {job_id}")
        traces = data_manipulation.readin_traces(
            data_dir, patient_file_path, start_index=100
        )

        if filter_age is not None:
            traces = [t for t in traces if int(t.sub_age.item()) < filter_age]
         
        if verbose:
            print(f"Number of patient traces: {len(traces)}")

        # Convert the images to Eye trace features 
        if ms_only:
            X, y, subject_ids = pre_processing.build_using_edss(traces)

            # Augment Severe EDSS indices to get rid of data imbalance
            severe_indices = np.where(y == 1)
            X_aug, y_aug, sub_id_aug = pre_processing.flip_augment(X, y, 
                                                                   subject_ids,
                                                                   severe_indices)
            # Append these Augmented samples 
            print(subject_ids[severe_indices].shape)
            X = np.concatenate((X, X_aug))
            y = np.concatenate((y, y_aug))
            subject_ids = np.concatenate((subject_ids, sub_id_aug))
                 
        else:
            X, y, subject_ids = pre_processing.traces_to_feature(traces, 
                                                                 mean_center=False, scale_std=False)

        # Use only distance and test it out 
        # X = pre_processing.use_dist(X)

        # Prepare the Data. Augment the data.  
        X = pre_processing.downsample_traces(X, 512)
        
        # Use spectrograms if use_spectrogram is set to True
        if use_spectrogram:
            X = pre_processing.spec_build(X)
            freq_bins = X.shape[2]
            time_sets = X.shape[3]
            X = np.reshape(X, newshape=(-1, 2, freq_bins * time_sets))

        if verbose:
            print(f"Number of samples with Label  0:{y[y == False].shape[0]}")
            print(f"Number of samples with Label  1:{y[y == True].shape[0]}")
            print(f"Number of features used from Eye trace: {X.shape[1]}")
        
        # Create the test train split here 
        (
            X_train,
            X_test,
            y_train,
            y_test,
            subject_ids_train,
            subject_ids_test,
        ) = grouped_train_test_split(
            X, y, subject_ids, test_fraction, random_seed=random_seed
        )

        if verbose:
            print(f"Shape of the Input is : {X_train.shape}")
        
        # Double the dataset by including the flipped version of the inputs
        if flip_y:
            X_train, y_train, subject_ids_train = data_manipulation.flip_y_traces(
                X_train, y_train, subject_ids_train
            )

        X_train = data_manipulation.channel_last(X_train)
        X_test = data_manipulation.channel_last(X_test)
        y_train, y_test = (
            y_train.astype(np.int32).squeeze(),
            y_test.astype(np.int32).squeeze(),
        )
        
        if verbose:
            print(f"Shape of the Input after setting channel last is {X_train.shape}")
        
        input_shape = tuple(list(X_train.shape)[1:])
        print(f"Shape of the input is: {input_shape}") 
        
        if use_lstm:
            model = lstm_model.create_model(input_shape)    
            
            if verbose:
                model.summary()
            
            lstm_model.run_model(model, X_train, y_train, lr)
            
        else:
            # Build the Model 
            # Any new model should have this Keras Classifier wrapper.
            log_dir = log_dir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            model = KerasClassifier(
                resnet_build_fn_closure(input_shape, log_dir=log_dir,
                                        drop_rate=drop_rate), 
                batch_size=batch_size, 
                epochs=epochs)
        
            # How many input channels 
            in_channel = X_train.shape[2]
            if verbose:
                print(f"Number of Input Channels: {in_channel}")
                print(f"Shape of X : {X.shape}")
                print(f"Shape of Y: {y.shape}")
            # Parameters that are used for the model 
            param_grid = {
                "num_layers": [num_layers],
                "n_feature_maps": [in_channel],
                "lr": [lr],
                "early_stopping": [earl_stopping],
                "kernel_multiple": [1],
            }

            cv_results = manual_grid_search_with_validation(
                X_train, y_train, subject_ids_train, 
                num_folds, model, param_grid,
                model_path, model_name)
        
        # TODO: Use the X_test set to calculate some metrics 
        y_pred = model.predict_proba(X_test)
        test_accuracy = accuracy_score(y_test, np.argmax(y_pred, axis=1))

        print(cv_results)
        save_path = Path("~").expanduser() / "cv_runs"
        # Create the directory if it doesn't exist
        save_path.mkdir(parents=True, exist_ok=True)
        save_file = save_path / f"cv_run_{job_id}.csv"
        cv_results.to_csv(save_file)

        # Test Accuracy Calculation
        print(f"Test Accuracy of the model is: {test_accuracy:.2f} when tested on {X_test.shape[0]} samples.")


if __name__ == "__main__":
    fire.Fire(Train)
