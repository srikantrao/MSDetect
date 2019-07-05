import logging
import os
from pathlib import Path
import fire 
import uuid
import datetime
import sys
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
import pre_processing
import data_manipulation
import transforms

from models import resnet_build_fn_closure
from validation import (
    grouped_train_test_split,
    manual_grid_search_with_validation,
    grouped_accuracy,
    modeled_grouped_accuracy,
    manual_grid_search_with_validation_sklearn,
    ordinal_grid_search,
    grouped_metrics,
)

import lstm_model

# Set numpy print options 
np.set_printoptions(2)

# Show only Error logs for Tensorflow 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Probably used to work around this Error - This means that multiple copies of the OpenMP runtime have been linked into the program
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


logger = logging.getLogger(__name__)

class Tasks: 
    def ordinal(
        self,
        data_dir="/data/envision_working_traces/",
        patient_file_path="../data/updated_patients_stats.csv",
        num_folds=10,
        num_splits=1,
        random_seed=666,
        test_fraction=0.15,
        flip_y=False,
        filter_age=None,
    ):
        data_dir = Path(data_dir).absolute()
        patient_file_path = Path(patient_file_path).absolute()
        job_id = str(uuid.uuid4())
        logger.info(f"Job ID for this run: {job_id}")
        trials = data_manipulation.load_augmented_trials(
            data_dir,
            patient_file_path,
            filter_age=filter_age,
            flip_y=flip_y,
            num_splits=num_splits,
            just_patients=True,
        )

        steps = [
            ("ffts", transforms.FFTFeatures(60)),
            ("velocities", transforms.MeanVelocity()),
            ("pca", transforms.PCATransformer()),
        ]
        pipeline = Pipeline([("features", FeatureUnion(steps))])
        y = np.array([float(t.sub_edss.item()) for t in trials])
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

        subject_ids = np.array([t.subjid for t in trials])

        gss = GroupShuffleSplit(
            n_splits=1, test_size=test_fraction, random_state=random_seed
        )
        for train_index, test_index in gss.split(trials, y, subject_ids):
            train_trials = [trials[idx] for idx in train_index]
            # test_trials = [trials[idx] for idx in test_index]

            y_train = y[train_index]
            # y_test = y[test_index]

            subject_ids_train = subject_ids[train_index]
            # subject_ids_test = subject_ids[test_index]

            X_train = pipeline.fit_transform(train_trials)
            # X_test = pipeline.transform(test_trials)

        model = Pipeline([("model", LogisticIT())])
        param_grid = {"model__alpha": [0, 1, 10, 100]}
        results = ordinal_grid_search(
            X_train,
            y_train,
            subject_ids_train,
            num_folds,
            model,
            param_grid,
            label_encoder,
        )
        results = results.sort_values(["mean_val_grouped_mae"], ascending=True)
        print(results)
        best_params = results[list(param_grid.keys())].iloc[0].to_dict()

        model = model.set_params(**best_params)

        model.fit(X_train, y_train)

    def xgboost_classification(
        self,
        cross_validate=False,
        learning_curve=False,
        model_trace_reduction=False,
        age_distribution=False,
        data_dir="/data/envision_working_traces/",
        patient_file_path="./data/patient_stats.csv",
        num_folds=10,
        num_splits=1,
        random_seed=666,
        test_fraction=0.25,
        flip_y=False,
        filter_age=None,
    ):
        """
        Train an xgboost model to predict patient/control.

        This function includes a number of optional flags to go beyond simply
        training the model. See the Parameters for more information.

        Parameters
        ----------
        cross_validate
            Whether or not to perform cross validation over a range of
            hyperparameters. The hyperparameters are hardcoded.
        learning_curve
            Whether or not to generate a learning curve which shows how the
            model performance changes as a function of number of subjects.
            Results are saved in a file called learning_curve.csv
        model_trace_reduction
            Whether or not to use a logistic regression model for reducing
            a subject's multiple trace predictions into a single
            subject-level prediction.
        age_distribution
            Whether or not to generate a distribution of performance vs.
            subject age.
        data_dir
            Directory that contains the subject traces
        patient_file_path
            File that contains information about each subject
        num_folds
            Number of cross validation folds for k-fold cross validation
        num_splits
            Number of splits to break up each trace into
        random_seed
            Random seed to use in splitting data
        test_fraction
            Fraction of data to reserve for the test set
        flip_y
            Whether or not to duplicate the trace data by flipping the y trace
        filter_age
            If included filter out any subjects whose age is >= the filter_age

        """
        # The below parameters will be used during all training except for
        # cross validation (in that case, we will do a hyperparameter sweep).
        xgboost_params = {"n_jobs": -1, "n_estimators": 10, "reg_lambda": 1}

        job_id = str(uuid.uuid4())
        logger.info(f"Job ID for this run: {job_id}")
        trials = data_manipulation.load_augmented_trials(
            data_dir,
            patient_file_path,
            filter_age=filter_age,
            flip_y=flip_y,
            num_splits=num_splits,
        )

        ffts = data_manipulation.get_fft_information(trials, 60)
        velocities = data_manipulation.get_vel_information(trials)

        X = np.hstack((ffts, velocities))

        y = np.array([t.is_patient for t in trials])

        subject_ids = np.array([t.subjid for t in trials])

        gss = GroupShuffleSplit(
            n_splits=1, test_size=test_fraction, random_state=random_seed
        )
        for train_index, test_index in gss.split(X, y, subject_ids):
            train_trials = [trials[idx] for idx in train_index]
            test_trials = [trials[idx] for idx in test_index]
            pca_train, pca = data_manipulation.pca_whiten_reduce(
                train_trials, 0.99
            )
            pca_test, _ = data_manipulation.pca_whiten_reduce(
                test_trials, pca=pca
            )

            X_train = X[train_index]
            X_train = np.hstack((X_train, pca_train))

            X_test = X[test_index]
            X_test = np.hstack((X_test, pca_test))

            y_train = y[train_index]
            y_test = y[test_index]

            subject_ids_train = subject_ids[train_index]
            subject_ids_test = subject_ids[test_index]

        if cross_validate:

            param_grid = {
                "n_estimators": [1, 2, 5, 10, 20],
                "reg_lambda": [1, 10, 100, 1000],
            }

            cv_results = manual_grid_search_with_validation_sklearn(
                X_train,
                y_train,
                subject_ids_train,
                num_folds,
                XGBClassifier(n_jobs=-1),
                #             DecisionTreeClassifier(),
                param_grid,
                use_model=model_trace_reduction,
            )

            print(
                cv_results.sort_values(
                    ["mean_val_grouped_accuracy"], ascending=False
                )
            )

        if learning_curve:
            # Learning Curve
            fractions = range(10, 101, 1)
            unique_subjects = np.unique(subject_ids_train)
            all_acc_mean = []
            all_acc_std = []
            num_train_subjects = []
            for fraction in fractions:
                accs = []
                sens = []
                spec = []
                size = int(fraction / 100 * len(unique_subjects))

                for fold in range(100):
                    train_subjects = np.random.choice(
                        unique_subjects, size=size, replace=False
                    )

                    model = XGBClassifier(**xgboost_params)
                    idxs = np.in1d(subject_ids_train, train_subjects)
                    model.fit(X_train[idxs], y_train[idxs])
                    y_pred = model.predict_proba(X_test)
                    accuracy, sensitivity, specificity, _ = grouped_metrics(
                        y_test, y_pred, subject_ids_test, threshold=0.5
                    )
                    accs.append(accuracy)
                    sens.append(sensitivity)
                    spec.append(specificity)

                print(f"Size: {size}, fraction: {fraction}%")
                print(
                    f"Test accuracy {np.mean(accs):4.2f} "
                    f"(+/- {np.std(accs):4.2f}), "
                    f"sensitivity {np.mean(sens):4.2f} "
                    f"(+/- {np.std(sens):4.2f}), "
                    f"specificity {np.mean(spec):4.2f} "
                    f"(+/- {np.std(spec):4.2f}). (threshold = 0.5)"
                )
                all_acc_mean.append(np.mean(accs))
                all_acc_std.append(np.std(accs))
                num_train_subjects.append(size)

            pd.DataFrame(
                {
                    "mean_test_accuracy": all_acc_mean,
                    "std_test_accuracy": all_acc_std,
                    "num_train_subjects": num_train_subjects,
                }
            ).to_csv("learning_curve.csv", index=False)

        # Train on full training set, then test metrics

        model = XGBClassifier(**xgboost_params)

        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_train)
        (accuracy, sensitivity, specificity, metric_model) = grouped_metrics(
            y_train,
            y_pred,
            subject_ids_train,
            threshold=0.5,
            use_model=model_trace_reduction,
        )

        y_pred = model.predict_proba(X_test)

        # Try out two different thresholds:

        (accuracy, sensitivity, specificity, metric_model) = grouped_metrics(
            y_test,
            y_pred,
            subject_ids_test,
            threshold=0.5,
            use_model=model_trace_reduction,
            model=metric_model,
        )
        print(
            f"Test accuracy {accuracy:4.2f}, sensitivity {sensitivity:4.2f}, "
            f"specificity {specificity:4.2f}. (threshold = 0.5)"
        )

        threshold = 0.43
        (accuracy, sensitivity, specificity, metric_model) = grouped_metrics(
            y_test,
            y_pred,
            subject_ids_test,
            threshold=threshold,
            use_model=model_trace_reduction,
            model=metric_model,
        )
        print(
            f"Test accuracy {accuracy:4.2f}, sensitivity {sensitivity:4.2f}, "
            f"specificity {specificity:4.2f}. (threshold = {threshold})"
        )

        if age_distribution:
            # Age distribution
            ages = np.array([int(t.sub_age.item()) for t in test_trials])
            bins = [10, 20, 30, 40, 55]
            for b1, b2 in zip(bins[:-1], bins[1:]):
                idxs = np.where((ages >= b1) & (ages < b2))[0]
                acc = grouped_accuracy(
                    y_test[idxs],
                    y_pred[idxs],
                    subject_ids_test[idxs],
                    threshold=0.44,
                )
                print(f"Age {b1}-{b2} accuracy: {acc:4.2f}")
                baseline = np.zeros_like(y_pred[idxs])
                baseline[:, 1] = 1.0
                acc = grouped_accuracy(
                    y_test[idxs],
                    baseline,
                    subject_ids_test[idxs],
                    threshold=0.44,
                )
                print(f"\tbase accuracy {acc:4.2f}")
 
    def keras_classification(
        self,
        data_dir = "/data/envision_working_traces/",
        patient_file_path = "../data/patient_stats.csv",
        model_path = "./output",
        model_name = "",
        num_layers = 3,
        lr = 0.00005,
        drop_rate = 0.2,
        num_folds = 1,
        num_splits = 1,
        flip_y = False,
        random_seed = None,
        test_fraction = 0.1,
        filter_age = None,
        batch_size = 32,
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
                "early_stopping": [False],
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
    fire.Fire(Tasks)
