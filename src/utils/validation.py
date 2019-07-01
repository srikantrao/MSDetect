"""
Functions to help with calculating validation metrics. Because a lot of the
work in this repo involves _grouped_ metrics (i.e. there are multiple traces
 per subject), we often have to reimplement scikit-learn validation routines
 manually.
"""

import collections
import logging
import sys
from typing import Dict, Optional, Tuple, Union
import os 
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
)
from sklearn.model_selection import (
    GroupShuffleSplit,
    ParameterGrid,
    GroupKFold,
)
from sklearn.preprocessing import LabelEncoder
from tqdm.autonotebook import tqdm
from tabulate import tabulate 

logger = logging.getLogger(__name__)


def grouped_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    threshold: float = 0.5,
) -> np.float32:
    """
    Calculate accuracy when samples should be grouped together. A group
    prediction is taken to be the average prediction amongst the sample-level
    predictions within that group.
    
    Parameters
    ----------
    y_true
        True labels, shape = [num_samples,], Each entry is an integer
        referencing the true class label.
    y_pred
        Predicted class probabilities, shape = [num_samples, num_classes]
    groups
        Array which identifies which group each sample belongs to,
        shape = [num_samples,]
    threshold
        Minimum prediction probability for applying the class label.
    
    Returns
    -------
    accuracy_score
        The prediction accuracy
    """
    # Create a Group to Indices Dict 
    group_to_idxs = collections.defaultdict(list)
    for idx, group in enumerate(groups):
        group_to_idxs[group].append(idx)

    grouped_true = []
    grouped_pred = []
    for group, idxs in group_to_idxs.items():
        these_true = (np.mean(y_true[idxs]) > 0.5).astype(np.int32)
        these_pred = y_pred[idxs, :]
        grouped_true.append(these_true)
        grouped_pred.append(these_pred.mean(axis=0)[1] > threshold)
    return accuracy_score(grouped_true, grouped_pred)


def modeled_grouped_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    lr: Optional[LogisticRegression] = None,
) -> np.float32:
    """
    Calculate accuracy when samples should be grouped together. A logistic
    regression model is used to fit/predict a group prediction from the
    sample-level predictions within that group. The features of the model are
    the prediction mean, standard deviation, and sum of squares.

    Parameters
    ----------
    y_true
        True labels, shape = [num_samples,], Each entry is an integer referencing
        the true class label.
    y_pred
        Predicted class probabilities, shape = [num_samples, num_classes]
    groups
        Array which identifies which group each sample belongs to, shape = [num_samples,]
    lr
        Logistic regression model which is used for generating a prediction
        from a group of predictions. If not included, then a new model is fit.
        If included, then the existing model is used for prediction.
    
    Returns
    -------
    accuracy_score
        The prediction accuracy
    """
    group_to_idxs = collections.defaultdict(list)
    for idx, group in enumerate(groups):
        group_to_idxs[group].append(idx)

    grouped_true = []
    grouped_pred = []
    for group, idxs in group_to_idxs.items():
        these_true = (np.mean(y_true[idxs]) > 0.5).astype(np.int32)
        these_pred = y_pred[idxs, :]
        grouped_true.append(these_true)
        mean_feature = these_pred.mean(axis=0)[1]
        sum_sq_feature = (these_pred ** 2).mean(axis=0)[1]
        std_features = these_pred.std(axis=0)[1]
        grouped_pred.append([mean_feature, sum_sq_feature, std_features])

    grouped_pred = np.array(grouped_pred)
    if lr is None:
        lr = LogisticRegression(solver="lbfgs")
        lr.fit(grouped_pred, grouped_true)
    grouped_pred = lr.predict(grouped_pred)
    return accuracy_score(grouped_true, grouped_pred), lr


def max_grouped_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    thresholds: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Determine the maximum grouped accuracy when sweeping across an array of
    thresholds. Returns the maximum accuracy and the threshold for the max
    accuracy.
    """
    accuracies = []
    for t in thresholds:
        accuracies.append(
            grouped_accuracy(y_true, y_pred, groups, threshold=t)
        )
    accuracies = np.array(accuracies)
    best_idx = np.argmax(accuracies)
    return accuracies[best_idx], thresholds[best_idx]


def grouped_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    threshold: float = 0.5,
    use_model: bool = False,
    model: Optional[LogisticRegression] = None,
) -> Tuple[np.float32, np.float32, np.float32, Optional[LogisticRegression]]:
    """
    Calculate grouped accuracy, sensitivity, and specificity.

    Parameters
    ----------
    y_true
        True labels
    y_pred
        Prediction scores
    groups
        Which group y_true and y_pred belong to
    threshold
        Threshold to use in converting from a continuous prediction score to a
        predicted label.
    use_model
        Whether or not to use a model to combine subject-level predictions into
        a single group score
    model
        The model to use for generating group scores. If not included, then fit
        a new logistic regression.

    Returns
    -------
    accuracy
    sensitivity
    specificity
    model

    """
    group_to_idxs = collections.defaultdict(list)
    for idx, group in enumerate(groups):
        group_to_idxs[group].append(idx)

    grouped_true = []
    grouped_pred = []
    for group, idxs in group_to_idxs.items():
        these_true = (np.mean(y_true[idxs]) > 0.5).astype(np.int32)
        these_pred = y_pred[idxs, :]

        if use_model:
            mean_feature = these_pred.mean(axis=0)[1]
            sum_sq_feature = (these_pred ** 2).mean(axis=0)[1]
            std_features = these_pred.std(axis=0)[1]
            these_pred = [mean_feature, sum_sq_feature, std_features]
        else:
            these_pred = these_pred.mean(axis=0)[1] > threshold

        grouped_true.append(these_true)
        grouped_pred.append(these_pred)

    if use_model:
        grouped_pred = np.array(grouped_pred)
        if model is None:
            model = LogisticRegression(solver="lbfgs")
            model.fit(grouped_pred, grouped_true)
        grouped_pred = model.predict_proba(grouped_pred)[:, 1] > threshold

    accuracy = accuracy_score(grouped_true, grouped_pred)
    tn, fp, fn, tp = confusion_matrix(grouped_true, grouped_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return accuracy, sensitivity, specificity, model


def grouped_ordinal_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    label_encoder: LabelEncoder,
) -> Tuple[np.float32, np.float32]:
    """
    Calculate ordinal-specific metrics (accuracy and mean absolute error) for
    grouped data.

    Parameters
    ----------
    y_true
        True labels
    y_pred
        Prediction scores
    groups
        Which group y_true and y_pred belong to
    label_encoder
        Used for converting from an ordinal score to a multiclass label and
        back again.

    Returns
    -------
    accuracy
        Group level accuracy
    mae
        Group level mean absolute error

    """
    group_to_idxs = collections.defaultdict(list)
    for idx, group in enumerate(groups):
        group_to_idxs[group].append(idx)

    grouped_true = []
    grouped_pred = []
    for group, idxs in group_to_idxs.items():
        these_true = np.max(y_true[idxs]).astype(np.int32)

        these_pred = y_pred[idxs, :].sum(axis=0)
        these_pred = softmax(these_pred)

        grouped_true.append(these_true)
        grouped_pred.append(these_pred)

    grouped_true = np.array(grouped_true)
    grouped_pred = np.argmax(np.vstack(grouped_pred), axis=1)

    accuracy = accuracy_score(grouped_true, grouped_pred)

    grouped_true = label_encoder.inverse_transform(grouped_true)
    grouped_pred = label_encoder.inverse_transform(grouped_pred)

    mae = mean_absolute_error(grouped_true, grouped_pred)
    return accuracy, mae


def max_grouped_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    thresholds: np.ndarray,
) -> Tuple[np.float32, np.float32, np.float32, np.float32]:
    """
    Calculated maximum grouped metrics by sweeping an array of classification
    thresholds.
    """
    accuracies = []
    sensitivities = []
    specificities = []
    for t in thresholds:
        accuracy, sensitivity, specificity, _ = grouped_metrics(
            y_true, y_pred, groups, threshold=t
        )
        accuracies.append(accuracy)
        sensitivities.append(sensitivity)
        specificities.append(specificity)

    accuracies = np.array(accuracies)
    sensitivities = np.array(sensitivities)
    specificities.append(specificity)

    best_idx = np.argmax(accuracies)
    return (
        accuracies[best_idx],
        sensitivities[best_idx],
        specificities[best_idx],
        thresholds[best_idx],
    )


def grouped_train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    test_fraction: float,
    random_seed: Optional[int] = None,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """
    Split training data, (X, y) into training and test data. Each sample of the
    data belongs to a group, as defined by `groups`. This function makes sure
    that the groups in training and test are disjoint (no sample from a group in
    training appears in test, and vice versa)
    
    Parameters
    ----------
    X
        The training input data, shape = [num_samples, ...] (there can be
        multiple other dimensions,
        depending on the training data)
    y
        The class labels, shape = [num_samples,] with values of True (
        False) if subject is a patient (is a control).
    groups
        Array of subject ids / groups that each sample belongs to,
        shape = [num_samples,]
    test_fraction
        The fraction of the training groups to move to the test data.
        Should be a number between 0 and 1.
    random_seed
        Random seed to use when splitting the data.
    
    """
    gss = GroupShuffleSplit(
        n_splits=1, test_size=test_fraction, random_state=random_seed
    )
    for train_index, test_index in gss.split(X, y, groups):
        X_train, y_train = X[train_index], y[train_index]
        groups_train = groups[train_index]

        X_test, y_test = X[test_index], y[test_index]
        groups_test = groups[test_index]

    return X_train, X_test, y_train, y_test, groups_train, groups_test


def manual_grid_search_with_validation(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    num_folds: int,
    model,
    param_grid: Dict[str, list],
    model_path,
    model_name ) -> pd.DataFrame:
    """
    Run a grid search and collect grouped validation metrics. This function 
    implements K-fold cross validation, where each fold contains disjoint
    groups.
    
    Returns a dataframe with the result of the cross validation.
    
    Ideally, we would use scikit_learn.model_selection.GridSearchCV instead of 
    manually running a grid search. However, due to some peculiarities of
    also using keras (e.g. we want to pass validation data into model.fit()
    for early stopping), it ended up being easier to just write this
    function than hacking GridSearchCV.
    
    Parameters
    ----------
    X
        Training data
    y
        Class labels 
    groups
        Labels identifying which group each sample in (X, y) belongs to
    num_folds
        Number of folds to do for K-fold cross validation
    model
        Classification model
    param_grid
        Grid of hyperparameters to search in this function. Format should be
        key = hyperparameter_name, value = list of hyperparameter values to
        search.
    model_path 
        Folder where the model should be saved.
    model_name
        Name of the model. Will get saved as model_name.h5
        
    Returns
    -------
    cv_results
        DataFrame of results with columns:
        mean_train_accuracy, std_train_accuracy,
        mean_val_accuracy, std_val_accuracy,
        mean_train_grouped_accuracy, std_train_grouped_accuracy,
        mean_val_grouped_accuracy, std_val_grouped_accuracy,
        
        and then a column for each hyperparameter
        
        Each row is the results from a different set of hyperparameters,
        aggregated over the experiment folds.
    """
    
    # Generates all the combinations of the different values provided 
    param_grid = ParameterGrid(param_grid)
    cv_results = []

    for params in param_grid:
        # Store all the final results of this fold in this dict
        fold_results = {
            "train_accuracy": [],
            "val_accuracy": [],
            "train_grouped_accuracy": [],
            "val_grouped_accuracy": [],
        }
        # Generate the train and test indices for the 
        # different splits. 
        
        #  GroupKFold makes sure that you are samples from the same group
        # do not end up in both the test set and the train set.
        
        # Modify this so that Group Folding is not used 
        # for the simple experiments 
        if num_folds > 1:
            logger.info("Group K Fold split is used since num_folds is greater than 1")
            sf = GroupKFold(n_splits=num_folds).split(X, y, groups)
            logger.info("Current params:")
            logger.info(params)
        # If there is only one fold - Simple train, test, val 
        else:
            # Use Shuffle Split 
            logger.info("Shufle Split is used since only one split")
            sf  = GroupShuffleSplit(n_splits=1, test_size=.1, random_state=0).split(X, y, groups) 

        for fold, (train_index, val_index) in enumerate(sf):
            logger.info(f"Fitting fold {fold}")
            try:
                model.set_params(**params)
                # KerasClassifier wrapper has a function that implements fit.
                # Look at Sequential.fit to understand the different legal
                # parameters that can be passed.
                model.fit(
                    X[train_index],
                    y[train_index],
                    x_val=X[val_index],
                    y_val=y[val_index],
                )
                
                # Save the Model parameters
                model.model.save(model_name=model_name,
                                 model_path=model_path)
                print(f"File has been saved to {model_name}")

                # Calculate the Validation Accuracy 
                y_pred = model.predict_proba(X[val_index])
                val_accuracy = accuracy_score(
                    y[val_index],
                    # Convert from probability to a label
                    np.argmax(y_pred, axis=1),
                )
                # Calculate Val accuracy based on group 
                val_grouped_accuracy = grouped_accuracy(
                    y[val_index], y_pred, groups[val_index]
                )
                # Calculate the Training Accuracy
                y_pred = model.predict_proba(X[train_index])
                train_accuracy = accuracy_score(
                    y[train_index], np.argmax(y_pred, axis=1)
                )
                # Calculate the train accuracy based on group
                train_grouped_accuracy = grouped_accuracy(
                    y[train_index], y_pred, groups[train_index]
                )

                # Save the reults 
                fold_results["train_accuracy"].append(train_accuracy)
                fold_results["val_accuracy"].append(val_accuracy)
                fold_results["train_grouped_accuracy"].append(
                    train_grouped_accuracy
                )
                fold_results["val_grouped_accuracy"].append(
                    val_grouped_accuracy
                )
            except KeyboardInterrupt:
                logger.exception("Cancelling validation run", exc_info=False)
                sys.exit()
            except Exception:
                logger.exception("Error while fitting.")

        param_results = params.copy()
        # Average the values over all the folds 
        param_results["mean_train_accuracy"] = np.mean(
            fold_results["train_accuracy"]
        )
        param_results["mean_val_accuracy"] = np.mean(
            fold_results["val_accuracy"]
        )
        param_results["std_train_accuracy"] = np.std(
            fold_results["train_accuracy"]
        )
        param_results["std_val_accuracy"] = np.std(
            fold_results["val_accuracy"]
        )

        param_results["mean_train_grouped_accuracy"] = np.mean(
            fold_results["train_grouped_accuracy"]
        )
        param_results["mean_val_grouped_accuracy"] = np.mean(
            fold_results["val_grouped_accuracy"]
        )
        param_results["std_train_grouped_accuracy"] = np.std(
            fold_results["train_grouped_accuracy"]
        )
        param_results["std_val_grouped_accuracy"] = np.std(
            fold_results["val_grouped_accuracy"]
        )

        logger.info(f"Results from fold {fold}")
        logger.info(param_results)

        cv_results.append(param_results)
     
    return pd.DataFrame(cv_results)


def ordinal_grid_search(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    num_folds: int,
    model,
    param_grid: Dict[str, list],
    label_encoder: LabelEncoder,
) -> pd.DataFrame:
    """Similar to manual_grid_search_with_validation but for ordinal models"""
    param_grid = ParameterGrid(param_grid)
    cv_results = []

    for params in param_grid:
        fold_results = {
            "train_grouped_accuracy": [],
            "val_grouped_accuracy": [],
            "train_grouped_mae": [],
            "val_grouped_mae": [],
        }
        gkf = GroupKFold(n_splits=num_folds).split(X, y, groups)
        logger.info("Current params:")
        logger.info(params)
        pbar = tqdm(enumerate(gkf), total=num_folds)
        for fold, (train_index, val_index) in pbar:
            try:
                model.set_params(**params)
                model.fit(X[train_index], y[train_index])

                y_pred = model.predict_proba(X[train_index])
                (
                    train_grouped_accuracy,
                    train_grouped_mae,
                ) = grouped_ordinal_metrics(
                    y[train_index], y_pred, groups[train_index], label_encoder
                )

                y_pred = model.predict_proba(X[val_index])

                (
                    val_grouped_accuracy,
                    val_grouped_mae,
                ) = grouped_ordinal_metrics(
                    y[val_index], y_pred, groups[val_index], label_encoder
                )

                fold_results["train_grouped_accuracy"].append(
                    train_grouped_accuracy
                )
                fold_results["val_grouped_accuracy"].append(
                    val_grouped_accuracy
                )

                fold_results["train_grouped_mae"].append(train_grouped_mae)
                fold_results["val_grouped_mae"].append(val_grouped_mae)
                pbar.set_postfix(
                    fold=fold, grouped_val_acc=val_grouped_accuracy
                )

            except KeyboardInterrupt:
                logger.exception("Cancelling validation run", exc_info=False)
                sys.exit()

        param_results = params.copy()
        param_results["mean_train_grouped_accuracy"] = np.mean(
            fold_results["train_grouped_accuracy"]
        )
        param_results["mean_val_grouped_accuracy"] = np.mean(
            fold_results["val_grouped_accuracy"]
        )
        param_results["std_train_grouped_accuracy"] = np.std(
            fold_results["train_grouped_accuracy"]
        )
        param_results["std_val_grouped_accuracy"] = np.std(
            fold_results["val_grouped_accuracy"]
        )

        param_results["mean_train_grouped_mae"] = np.mean(
            fold_results["train_grouped_mae"]
        )
        param_results["mean_val_grouped_mae"] = np.mean(
            fold_results["val_grouped_mae"]
        )
        param_results["std_train_grouped_mae"] = np.std(
            fold_results["train_grouped_mae"]
        )
        param_results["std_val_grouped_mae"] = np.std(
            fold_results["val_grouped_mae"]
        )
    

        cv_results.append(param_results)
    
    # Create a DataFrame and then pretty print it 
    df = pd.DataFrame(cv_results)
    print(tabulate(df, headers='keys', tablefmt='psql'))
    
    return pd.DataFrame(cv_results)


def manual_grid_search_with_validation_sklearn(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    num_folds: int,
    model,
    param_grid: Dict[str, list],
    use_model=False,
) -> pd.DataFrame:
    """
    Run a grid search and collect grouped validation metrics. This function 
    implements K-fold cross validation, where each fold contains disjoint
    groups.
    
    Returns a dataframe with the result of the cross validation.
    
    Ideally, we would use scikit_learn.model_selection.GridSearchCV instead of 
    manually running a grid search. However, due to some peculiarities of
    also using keras (e.g. we want to pass validation data into model.fit()
    for early stopping), it ended up being easier to just write this
    function than hacking GridSearchCV.
    
    Parameters
    ----------
    X
        Training data
    y
        Class labels 
    groups
        Labels identifying which group each sample in (X, y) belongs to
    num_folds
        Number of folds to do for K-fold cross validation
    model
        Classification model
    param_grid
        Grid of hyperparameters to search in this function. Format should be
        key = hyperparameter_name, value = list of hyperparameter values to
        search.
    use_model
        Whether or not to use a model to aggregate subject-level scores into
        group-level scores
        
    Returns
    -------
    cv_results
        DataFrame of results with columns:
        mean_train_accuracy, std_train_accuracy,
        mean_val_accuracy, std_val_accuracy,
        mean_train_grouped_accuracy, std_train_grouped_accuracy,
        mean_val_grouped_accuracy, std_val_grouped_accuracy,
        
        and then a column for each hyperparameter
        
        Each row is the results from a different set of hyperparameters,
        aggregated over the experiment folds.
    """
    param_grid = ParameterGrid(param_grid)
    cv_results = []

    for params in param_grid:

        fold_results = {
            "train_accuracy": [],
            "val_accuracy": [],
            "train_grouped_accuracy": [],
            "val_grouped_accuracy": [],
            "train_grouped_sensitivity": [],
            "val_grouped_sensitivity": [],
            "train_grouped_specificity": [],
            "val_grouped_specificity": [],
        }

        gkf = GroupKFold(n_splits=num_folds).split(X, y, groups)
        logger.info("Current params:")
        logger.info(params)
        pbar = tqdm(enumerate(gkf), total=num_folds)
        for fold, (train_index, val_index) in pbar:
            try:
                model.set_params(**params)
                model.fit(X[train_index], y[train_index])

                y_pred = model.predict_proba(X[train_index])
                train_accuracy = accuracy_score(
                    y[train_index], np.argmax(y_pred > 0.5, axis=1)
                )
                (
                    train_grouped_accuracy,
                    train_grouped_sensitivity,
                    train_grouped_specificity,
                    metric_model,
                ) = grouped_metrics(
                    y[train_index],
                    y_pred,
                    groups[train_index],
                    use_model=use_model,
                )

                y_pred = model.predict_proba(X[val_index])
                val_accuracy = accuracy_score(
                    y[val_index],
                    # Convert from probability to a label
                    np.argmax(y_pred > 0.5, axis=1),
                )
                (
                    val_grouped_accuracy,
                    val_grouped_sensitivity,
                    val_grouped_specificity,
                    metric_model,
                ) = grouped_metrics(
                    y[val_index],
                    y_pred,
                    groups[val_index],
                    use_model=use_model,
                    model=metric_model,
                )

                fold_results["train_accuracy"].append(train_accuracy)
                fold_results["val_accuracy"].append(val_accuracy)

                fold_results["train_grouped_accuracy"].append(
                    train_grouped_accuracy
                )
                fold_results["val_grouped_accuracy"].append(
                    val_grouped_accuracy
                )

                fold_results["train_grouped_sensitivity"].append(
                    train_grouped_sensitivity
                )
                fold_results["val_grouped_sensitivity"].append(
                    val_grouped_sensitivity
                )

                fold_results["train_grouped_specificity"].append(
                    train_grouped_specificity
                )
                fold_results["val_grouped_specificity"].append(
                    val_grouped_specificity
                )
                pbar.set_postfix(
                    fold=fold, grouped_val_acc=val_grouped_accuracy
                )

            except KeyboardInterrupt:
                logger.exception("Cancelling validation run", exc_info=False)
                sys.exit()
            except Exception:
                logger.exception("Error while fitting.")

        param_results = params.copy()
        param_results["mean_train_accuracy"] = np.mean(
            fold_results["train_accuracy"]
        )
        param_results["mean_val_accuracy"] = np.mean(
            fold_results["val_accuracy"]
        )
        param_results["std_train_accuracy"] = np.std(
            fold_results["train_accuracy"]
        )
        param_results["std_val_accuracy"] = np.std(
            fold_results["val_accuracy"]
        )

        param_results["mean_train_grouped_accuracy"] = np.mean(
            fold_results["train_grouped_accuracy"]
        )
        param_results["mean_val_grouped_accuracy"] = np.mean(
            fold_results["val_grouped_accuracy"]
        )
        param_results["std_train_grouped_accuracy"] = np.std(
            fold_results["train_grouped_accuracy"]
        )
        param_results["std_val_grouped_accuracy"] = np.std(
            fold_results["val_grouped_accuracy"]
        )

        param_results["mean_train_grouped_sensitivity"] = np.mean(
            fold_results["train_grouped_sensitivity"]
        )
        param_results["mean_val_grouped_sensitivity"] = np.mean(
            fold_results["val_grouped_sensitivity"]
        )
        param_results["std_train_grouped_sensitivity"] = np.std(
            fold_results["train_grouped_sensitivity"]
        )
        param_results["std_val_grouped_sensitivity"] = np.std(
            fold_results["val_grouped_sensitivity"]
        )

        param_results["mean_train_grouped_specificity"] = np.mean(
            fold_results["train_grouped_specificity"]
        )
        param_results["mean_val_grouped_specificity"] = np.mean(
            fold_results["val_grouped_specificity"]
        )
        param_results["std_train_grouped_specificity"] = np.std(
            fold_results["train_grouped_specificity"]
        )
        param_results["std_val_grouped_specificity"] = np.std(
            fold_results["val_grouped_specificity"]
        )

        cv_results.append(param_results)
    
    # Create a DataFrame and then pretty print it 
    df = pd.DataFrame(cv_results)
    print(tabulate(df, headers='keys', tablefmt='psql'))
    
    return pd.DataFrame(cv_results)
