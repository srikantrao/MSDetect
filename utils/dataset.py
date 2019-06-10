import numpy as np
import utils.data_formatutils as dfu

class Dataset(object):
    def __init__(self, data, lbls, stats, vectorize=False, rand_state=np.random.RandomState(None)):
        self.vectorize = vectorize
        self.data_shape = data.shape
        self.data = data
        self.stats = stats
        self.num_examples = self.get_num_examples()
        if self.vectorize:
            self.vectorize_data()
        self.labels = lbls
        self.rand_state = rand_state
        self.reset_counters()

    def get_num_examples(self):
        return self.data.shape[0]

    def reset_counters(self):
        """
          Reset all counters for batches & epochs completed
        """
        self.epochs_completed = 0
        self.batches_completed = 0
        self.batches_this_epoch = 0
        self.curr_epoch_idx = 0
        self.epoch_order = self.rand_state.permutation(self.num_examples)

    def new_epoch(self, num_to_advance=1):
        """
        Advance epoch counter & generate new index order
        Inputs:
          num_to_advance [int] number of epochs to advance
        """
        self.epochs_completed += int(num_to_advance)
        self.batches_this_epoch = 0
        for _ in range(int(num_to_advance)):
            self.epoch_order = self.rand_state.permutation(self.num_examples)

    def next_batch(self, batch_size):
        """
        Return a batch of data
        Outputs:
          3d tuple containing data, labels
        Inputs:
          batch_size [int] representing the number of data in the batch
            NOTE: If batch_size does not divide evenly into self.num_examples then
            some of the data will not be delivered. The function assumes that
            batch_size is a scalar increment of num_examples.
        """
        assert batch_size <= self.num_examples, (
            "Input batch_size was greater than the number of available examples.")
        if self.curr_epoch_idx + batch_size > self.num_examples:
            start = 0
            self.new_epoch(1)
            self.curr_epoch_idx = 0
        else:
            start = self.curr_epoch_idx
        self.batches_completed += 1
        self.batches_this_epoch += 1
        self.curr_epoch_idx += batch_size
        set_indices = self.epoch_order[start:self.curr_epoch_idx]
        if self.labels is not None:
            return (self.data[set_indices, ...], self.labels[set_indices, ...])
        return (self.data[set_indices, ...], self.labels)

    def advance_counters(self, num_batches, batch_size):
        """
        Increment member variables to reflect a step forward of num_batches data
        Inputs:
          num_batches: How many batches to step forward
          batch_size: How many examples constitute a batch
        """
        assert self.curr_epoch_idx == 0, ("Error: Current epoch index must be 0.")
        if num_batches * batch_size > self.num_examples:
            self.new_epoch(int((num_batches * batch_size) / float(self.num_examples)))
        self.batches_completed += num_batches
        self.curr_epoch_idx = (num_batches * batch_size) % self.num_examples

    def vectorize_data(self):
        """Reshape data to be a vector per data point"""
        self.data = self.data.reshape(self.num_examples, np.prod(self.data_shape[1:]))

    def devectorize_data(self):
        """Reshape images to be a vector per data point"""
        self.data = self.data.reshape(self.data_shape)

    def flip_trials_y(self):
        """Double dataset size by flipping all data on the y axis"""
        if self.vectorize:
            self.devectorize_data()
        flipped_data = dfu.flip_data_on_y_axis(self.data)
        self.data = np.concatenate((self.data, flipped_data), axis=0)
        self.data_shape = self.data.shape
        self.num_examples = self.get_num_examples()
        self.labels = np.concatenate((self.labels, self.labels), axis=0)
        self.stats = {key:np.concatenate((value, value), axis=0)
            for (key, value) in self.stats.items()}
        if self.vectorize:
            self.vectorize_data()

    def pca_whiten_reduce(self, explained_variance, rand_state=np.random.RandomState(None), pca=None):
        self.data, self.pca = dfu.pca_whiten_reduce(self.data, explained_variance, rand_state, pca)

    def concatenate_age_information(self):
        self.data = np.concatenate((self.data, self.stats["ages"]), axis=1)

    def concatenate_fft_information(self):
        self.data = np.concatenate((self.data, self.stats["ffts"]), axis=1)

    def concatenate_vel_information(self):
        self.data = np.concatenate((self.data, self.stats["velocities"]), axis=1)

    def concatenate_nblinks_information(self):
        self.data = np.concatenate((self.data, self.stats["nblinks"]), axis=1)

class DatasetGroup(object):
    """
    Class to hold a group of datasets, that are resampled multiple times
    for the purpose of cross validation
    Parameters:
        data: np.ndarray of shape [num_datapoints, ...]
        labels: np.ndarray of shape [num_datapoints] (where the int value indicates class)
        stats: dictionary containing keys {"ages", "ffts", "velocities", "nblinks"}
        params: class containing the following member variables:
          num_train: how many datapoints to use for training
          num_val: how many datapoints to use for validation
          num_test: how many datapoints to use for testing
          num_crossvalidations: how many crossvalidations to do
          vectorize: whether or not to vectorize the data
          rand_state: np.random.RandomState()
    TODO: if num_val or num_test is set to zero, the code might break. This should be fixed.
    """
    def __init__(self, data, lbls, stats, params):
        # pass on same params to individual child datasets
        self.data = data
        self.labels = lbls
        self.stats = stats
        self.vectorize = params.vectorize
        self.rand_state = params.rand_state
        self.num_train = params.num_train
        self.num_val = params.num_val
        self.num_test = params.num_test
        # keep our own set of summary params
        self.num_crossvalidations = params.num_crossvalidations
        self.data_list = self.create_dataset_list(self.num_crossvalidations, params)
        self.init_results()
        self.reset_counters()

    #TODO: iterate random state each time data are created so pull is different each time
    def create_dataset_list(self, num_datasets, params):
        """Create a list of datasets and labels"""
        data_list = []
        for i in range(num_datasets):
            if i == 0: # test set from initial draw will always be the test set
                data, labels, stats = dfu.format_mlp_data(self.data, self.labels, self.stats, params)
                # Pull out test data out for future draws
                test_data = data[2].copy()
                test_labels = labels[2].copy()
                test_stats = stats[2].copy()
                # Remove test data from object member variables
                self.data = np.concatenate(data[:2], axis=0)
                self.labels = np.concatenate([dfu.one_hot_to_dense(label)
                    for label in labels[:2]], axis=0)
                self.stats = dict()
                for ((key, val0), (_, val1)) in zip(stats[0].items(), stats[1].items()):
                  self.stats[key] = np.concatenate((val0, val1), axis=0)
            else:
                params.num_test = 0
                data, labels, stats = dfu.format_mlp_data(self.data, self.labels, self.stats, params)
            data = {
                "train": Dataset(data[0], labels[0], stats[0], self.vectorize, self.rand_state),
                "val": Dataset(data[1], labels[1], stats[1], self.vectorize, self.rand_state),
                "test": Dataset(test_data, test_labels, test_stats, self.vectorize, self.rand_state)}
            data_list.append(data)
        return data_list

    def get_dataset(self, idx):
        """Get one of the dataset elements"""
        return self.data_list[idx]

    def init_results(self):
        self.train_accuracies = np.zeros(self.num_crossvalidations)
        self.val_accuracies = np.zeros(self.num_crossvalidations)
        self.sensitivities = np.zeros(self.num_crossvalidations)
        self.specificities = np.zeros(self.num_crossvalidations)
        self.max_val_accuracy = np.zeros(self.num_crossvalidations)
        self.sens_at_max_acc = np.zeros(self.num_crossvalidations) #sens at max acc val
        self.spec_at_max_acc = np.zeros(self.num_crossvalidations) #spec at max acc val

    def reset_counters(self):
        self.crossvalids_completed = 0
        self.reset_child_counters()
    
    def reset_child_counters(self):
        for i in range(len(self.data_list)):
            self.data_list[i]['train'].reset_counters()
            self.data_list[i]['val'].reset_counters()
            self.data_list[i]['test'].reset_counters()

    def increment_counters(self):
        self.crossvalids_completed +=1
    
    def record_results(self, train_accuracy, val_accuracy, sensitivity, specificity, 
                       val_accuracy_max, sens_val_acc_max, spec_val_acc_max, idx):
        """ Record the results of a run"""
        self.train_accuracies[idx] = train_accuracy
        self.val_accuracies[idx] = val_accuracy
        self.sensitivities[idx] = sensitivity
        self.specificities[idx] = specificity
        self.max_val_accuracy[idx] = val_accuracy_max
        self.sens_at_max_acc[idx] = sens_val_acc_max
        self.spec_at_max_acc[idx] = spec_val_acc_max
        self.increment_counters()

    ## TODO: results start out as zero arrays. If results aren't all recorded, vals will be low.
    def mean_results(self):
        """Means for Results"""
        mean_train_accuracies = np.mean(self.train_accuracies)
        mean_val_accuracies = np.mean(self.val_accuracies)
        mean_sensitivities = np.mean(self.sensitivities)
        mean_specificities = np.mean(self.specificities)
        mean_maxacc = np.mean(self.max_val_accuracy)
        mean_sens_maxacc = np.mean(self.sens_at_max_acc)
        mean_spec_maxacc = np.mean(self.spec_at_max_acc)
        return(mean_train_accuracies, mean_val_accuracies, 
              mean_sensitivities, mean_specificities,
              mean_maxacc, mean_sens_maxacc, mean_spec_maxacc)

    def sd_results(self):
        """Standard Deviations for Results"""
        sd_train_accuracies = np.std(self.train_accuracies)
        sd_val_accuracies = np.std(self.val_accuracies)
        sd_sensitivities = np.std(self.sensitivities)
        sd_specificities = np.std(self.specificities)
        sd_maxacc = np.std(self.max_val_accuracy)
        sd_sens_maxacc = np.std(self.sens_at_max_acc)
        sd_spec_maxacc = np.std(self.spec_at_max_acc)
        return(sd_train_accuracies, sd_val_accuracies, 
              sd_sensitivities, sd_specificities,
              sd_maxacc, sd_sens_maxacc, sd_spec_maxacc)

    def flip_trials_y(self):
        for dataset in self.data_list:
            dataset["train"].flip_trials_y()
    
    def pca_whiten_reduce(self, explained_variance, rand_state):
        for index, dataset in enumerate(self.data_list):
            dataset["train"].pca_whiten_reduce(explained_variance, rand_state, pca=None)
            if index == 0:
                explained_variance = dataset["train"].pca.n_components
            dataset["val"].pca_whiten_reduce(explained_variance, rand_state,
                pca=dataset["train"].pca)
            dataset["test"].pca_whiten_reduce(explained_variance, rand_state,
                pca=dataset["train"].pca)

    def concatenate_age_information(self):
        for dataset in self.data_list:
            for key in dataset.keys():
              dataset[key].concatenate_age_information()

    def concatenate_fft_information(self):
        for dataset in self.data_list:
            for key in dataset.keys():
              dataset[key].concatenate_fft_information()

    def concatenate_vel_information(self):
        for dataset in self.data_list:
            for key in dataset.keys():
              dataset[key].concatenate_vel_information()

    def concatenate_nblinks_information(self):
        for dataset in self.data_list:
            for key in dataset.keys():
              dataset[key].concatenate_nblinks_information()
