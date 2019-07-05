"""
data_manipulation functions refactored into scikit-learn transforms. Most of
the code comes directly from the EnVision/utils/ module and is provided here
as is.
"""

from typing import List, Optional

import numpy as np
from scipy import signal
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_is_fitted

from utils.data_manipulation import EyeTrace


def rescale_data_to_one(data: np.ndarray) -> np.ndarray:
    """
    Rescales input data to be between 0 and 1
    TODO: Verify that this is actually the right thing to do?
        We want all of the data features to be on the same scale
        Should this normalization be done per sample or across the whole
        dataset?
    """
    data_min = np.amin(data)
    data_max = np.amax(data)
    return (data - data_min) / (data_max - data_min)


class FFTFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        super().__init__()

    def fit(
        self, trials: List[EyeTrace], y: Optional[np.ndarray] = None
    ) -> "FFTFeatures":
        return self

    def transform(
        self, trials: List[EyeTrace], y: Optional[np.ndarray] = None
    ) -> np.ndarray:

        """
        Extract fft of each trial, subsample to n_components
        Parameters
            trials: [list] with each element being an eyetrace object
            n_components: [int] number of FT components to subsample and add to data.
        TODO: give n_components a default
        """
        # calcualte Power Spectrums
        ft_x = [np.abs(np.fft.fft(trial.x)) ** 2 for trial in trials]
        ft_y = [np.abs(np.fft.fft(trial.y)) ** 2 for trial in trials]
        # subsample to number of desired components
        n_samples = self.n_components // 2
        # resample x and y fts, and concatenate them

        # NOTE: The line below generated a futures warning about indexing with
        # tuples. Don't have time to spend figuring out why, (it's soemthing
        # in the scipy.stats pacakage) but this code works for now.
        # the warning might go away with a future update of scipy.

        ss_fts = [
            np.concatenate(
                (
                    signal.resample(np.asarray(ft_x[i]), n_samples),
                    signal.resample(np.asarray(ft_y[i]), n_samples),
                )
            )
            for i in range(len(ft_x))
        ]
        # rescale
        ss_fts = rescale_data_to_one(ss_fts)
        return np.asarray(ss_fts)


class PCATransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        n_components_or_expl_variance=0.995,
        rand_state=np.random.RandomState(None),
    ):
        """
        Parameters
        ----------
        n_components_or_expl_variance: [float]
            indicating how many PCs should be used by requiring a minimum
            explained variance. 100*n_components_or_expl_variance = % variance
            explained of reduced data. Alternatively, if
            n_components_or_expl_variance is >= 1.0, then it is treated as an
            integer equal to the number of components to keep
        rand_state [np.random.RandomState()]

        """
        self.n_components_or_expl_variance = n_components_or_expl_variance
        self.rand_state = rand_state

    def fit(self, trials, y=None):
        # Reshape data to 2D if it is more than 2D
        minlen = np.min(
            [np.min((len(trial.x), len(trial.y))) for trial in trials]
        )
        # arrray is indexed by [tracenumber, timepoint, dim] where dim=2 is (x,y)
        trials_array = np.zeros((len(trials), minlen, 2))
        # fill in our array
        for i, trial in enumerate(trials):
            trials_array[i, :, 0] = trial.x[:minlen]
            trials_array[i, :, 1] = trial.y[:minlen]
        data = trials_array
        if data.ndim > 2:
            data = data.reshape(data.shape[0], np.prod(data.shape[1:]))
        if self.n_components_or_expl_variance >= 1.0:
            num_components = int(self.n_components_or_expl_variance)
        else:
            # Do full dim PCA to figure out how many components to keep
            pca = PCA(
                n_components=None,
                copy=True,
                whiten=False,
                svd_solver="auto",
                tol=0.0,
                iterated_power="auto",
                random_state=self.rand_state,
            )
            pca.fit(data)
            # Do reduced PCA based on original fit; return dim reduced data
            target_error = 1 - self.n_components_or_expl_variance
            num_components = np.max(
                np.argwhere(pca.explained_variance_ratio_ > target_error)
            )
        self.pca_ = PCA(
            n_components=num_components,
            copy=True,
            whiten=True,
            svd_solver="auto",
            tol=0.0,
            iterated_power="auto",
            random_state=self.rand_state,
        )
        self.pca_.fit(data)
        return self

    def _prepare_data(self, trials):
        # Reshape data to 2D if it is more than 2D
        minlen = np.min(
            [np.min((len(trial.x), len(trial.y))) for trial in trials]
        )
        # arrray is indexed by [tracenumber, timepoint, dim] where dim=2 is (x,y)
        trials_array = np.zeros((len(trials), minlen, 2))
        # fill in our array
        for i, trial in enumerate(trials):
            trials_array[i, :, 0] = trial.x[:minlen]
            trials_array[i, :, 1] = trial.y[:minlen]
        data = trials_array
        if data.ndim > 2:
            data = data.reshape(data.shape[0], np.prod(data.shape[1:]))
        return data

    def transform(self, trials, y=None):
        check_is_fitted(self, "pca_")
        data = self._prepare_data(trials)
        return self.pca_.transform(data)


class MeanVelocity(BaseEstimator, TransformerMixin):
    def fit(self, trials, y=None):
        return self

    def transform(self, trials, y=None):
        # calcualte Mean Velocity per trace
        vs = np.expand_dims(
            np.asarray([np.mean(np.asarray(trial.v)) for trial in trials]),
            axis=-1,
        )
        # rescale
        vs = rescale_data_to_one(vs)
        return vs
