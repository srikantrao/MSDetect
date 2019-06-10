import utils.data_formatutils as dfu
from utils.dataset import DatasetGroup
import sys
sys.path.append('../')

# Params
from params.model_params import params


class ThreeDatasets:
    def __init__(self, new_datasets_dict):
        """Creates the DatasetGroup objects to be fed into models for testing.
        Takes in dictionary of data at frequencies and creates
        DatasetGroup objects at the specified frequencies for each
        of three conditions:
                                - All data
                                - All data but Age
                                - Age only

        Args:
            new_datasets_dict: dictionary representing all eyetraces at
                               specified frequency in the form:
                                {freq1: [patient eyetraces, control eytraces],
                                 ...}
        Returns:
            Lists of dictionaries of DatasetGroup objects at each frequency,
            for each of the three testing conditions in the form:
                [all data, all data but age, age only], and
                 all data = {freq1: [[X_train, y_train], [X_test, y_test]],
                             ...}

        """
        # create dictionary for all the data with keys as frequencies
        # and values being DatasetGroups to be fed to the models
        self.all_data = {}
        for freq, data in new_datasets_dict.items():
            patient_trials = data[0]
            control_trials = data[1]
            if(len(patient_trials + control_trials) != 0):
                data, labels, stats = dfu.make_mlp_eyetrace_matrix(patient_trials, control_trials, params.fft_subsample)
                dataset_group_list = DatasetGroup(data, labels, stats, params)
                dataset_group_list.flip_trials_y()
                dataset_group_list.pca_whiten_reduce(params.required_explained_variance, params.rand_state)
                dataset_group_list.concatenate_age_information()
                dataset_group_list.concatenate_fft_information()
                dataset_group_list.concatenate_vel_information()
                dataset_group_list.concatenate_nblinks_information()

                X_train, y_train = (dataset_group_list.get_dataset(0)["train"].data,
                            dataset_group_list.get_dataset(0)["train"].labels)
                X_test, y_test = (dataset_group_list.get_dataset(0)["val"].data,
                            dataset_group_list.get_dataset(0)["val"].labels)
                self.all_data[freq] = [[X_train, y_train[:,1]], [X_test, y_test[:,1]]]

        # all data but age
        self.all_but_age = {}
        for freq, data in new_datasets_dict.items():
            patient_trials = data[0]
            control_trials = data[1]
            if(len(patient_trials + control_trials) != 0):
                data, labels, stats = dfu.make_mlp_eyetrace_matrix(patient_trials, control_trials, params.fft_subsample)
                dataset_group_list = DatasetGroup(data, labels, stats, params)
                dataset_group_list.flip_trials_y()
                dataset_group_list.pca_whiten_reduce(params.required_explained_variance, params.rand_state)
                # dataset_group_list.concatenate_age_information()
                dataset_group_list.concatenate_fft_information()
                dataset_group_list.concatenate_vel_information()
                dataset_group_list.concatenate_nblinks_information()

                X_train, y_train = (dataset_group_list.get_dataset(0)["train"].data,
                            dataset_group_list.get_dataset(0)["train"].labels)
                X_test, y_test = (dataset_group_list.get_dataset(0)["val"].data,
                            dataset_group_list.get_dataset(0)["val"].labels)
                self.all_but_age[freq] = [[X_train, y_train[:,1]], [X_test, y_test[:,1]]]

        # age only
        self.age_only = {}
        for freq, data in new_datasets_dict.items():
            patient_trials = data[0]
            control_trials = data[1]
            if(len(patient_trials + control_trials) != 0):
                data, labels, stats = dfu.make_mlp_eyetrace_matrix(patient_trials, control_trials, params.fft_subsample)
                dataset_group_list = DatasetGroup(data, labels, stats, params)
                dataset_group_list.concatenate_age_information()

                X_train, y_train = (dataset_group_list.get_dataset(0)["train"].data,
                            dataset_group_list.get_dataset(0)["train"].labels)
                X_test, y_test = (dataset_group_list.get_dataset(0)["val"].data,
                            dataset_group_list.get_dataset(0)["val"].labels)
                self.age_only[freq] = [[X_train[:,-1].reshape(-1,1), y_train[:,1]],
                                       [X_test[:,-1].reshape(-1,1), y_test[:,1]]]

    def get_three_datasets(self):
        return [self.all_data, self.all_but_age, self.age_only]
