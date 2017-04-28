
"""

    Author: Antonio Mejias Gil (anmegi.95@gmail.com)
    Date: Late 2016
    Description:
        Private module for dataset management. It can be used independently
        from the rest of the project.
        The capabilities, interface and error checking have been made according to
        good practices and a possible future opening in mind.
    
        Module functions include: dataset loading, target values encoding,
        normalisation, dataset splitting and mini-batch feeding.

"""

# Standard libraries
import csv
from random import shuffle
from enum import Enum, unique

# Third-party libraries
import numpy as np


@unique
class Problems(Enum):
    """
    Enumeration of problem types.
    Mostly for good coding practices and error checking (misspelt strings).
    """
    classification = 'classification'
    regression = 'regression'


@unique
class Encodings(Enum):
    """
    Enumeration of target values encoding.
    Mostly for good coding practices and error checking (misspelt strings).
    """
    # Regression problems and classification problems where the class is
    # expressed as an element of {1, 2, ... n_classes}
    numeric = 'numeric'
    # Classification problems, class expressed as vector (0,...0, 1, 0...0)
    one_hot = 'one-hot'

class DatasetException(Exception):

    def __init__(self, dataset, message):
        super(DatasetException, self).__init__('In dataset: {} -> {}'.format(dataset.name, message))


class Dataset:

    def __init__(self, path, problem, name='default'):
        """
        Returns a new object of class Dataset with numeric encoding.

        Regarding classification problem datasets: Classes will be encoded as an
        integer (use the encoding methods to change this). CSV files must have
        the target class as the last element in each line. If it is an integer,
        that value will be left. Otherwise a mapping between the found classes
        and integers will be created, which can be obtained with class_mapping.

        :param: path (string): Path to the source file
        :param: problem (string in {'classification', 'regression'}): Type of
                problem.
        :param: name (string): An optional name for the dataset. Default:
                'default'
        """

        # all attributes are listed here, with some dummy initialisations and
        # some meaningful ones. private attributes (nearly all of them) start
        # with _
        
        # basic features
        self.name = name
        self._problem = None
        self._encoding = Encodings.numeric
        self._normalisation = None
                
        # data
        self._n_examples, self.n_features = 0, 0
        self._ds_inputs, self._ds_targets = 0, 0
        
        # splitting
        # the subsets are saved as indices (with a random order), so that
        # the original data is untouched and several splittings are possible
        self._t_indices, self._v_indices, self._te_indices = [], [], []
        self._to_see_next = -1

        # in classification datasets
        self._mapping_dict = None
        self._classes = None

        inputs, targets = [], []

        try:
            self._problem = Problems.__members__[problem]
        except KeyError:
            raise DatasetException(self, 'Invalid problem type. Only "classification" and "regression" are currently supported')

        with open(path, 'rt') as file:
            source = csv.reader(file, delimiter=',')
            for row in source:
                inputs.append(row[:-1])
                targets.append(row[-1])

            if self._problem == Problems.classification:
                try:
                    # if the target values were integers, we leave them as they are
                    targets = [int(e) for e in targets]
                    self._mapping_dict = None
                    self._classes = sorted(list(set(targets)))
                except ValueError:
                    # otherwise we map them to indices and transform the targets
                    self._classes = list(set(targets))
                    self._mapping_dict = dict([(b, a) for (a, b) in enumerate(self._classes)])
                    targets = [self._mapping_dict[e] for e in targets]
                    self.ind_in_class = [[i for i in range(len(targets))
                                          if targets[i] == t] for t in range(len(self._classes))]

                self._n_classes = len(self._classes)
            else:
                targets = [float(e) for e in targets]

            self._ds_inputs = np.array(inputs, dtype=np.float)
            self._ds_targets = np.array(targets)

            self._n_examples, self._n_features = self._ds_inputs.shape

    def to_one_hot_encoding(self):
        """
        Changes the target value encoding to one-hot: 4 -> (0, 0, 0, 0, 1)
        Throws an exception if the problem is not a classification problem or it
        is not encoded with a suitable method to change to one-hot.

        :return: None

        """
        if self._problem != Problems.classification:
            raise DatasetException(self, 'Trying to change encoding on a dataset which does not belong to a classification problem')
        elif self._encoding != Encodings.numeric:
            raise DatasetException(self, 'Trying to apply one-hot encoding to a dataset which is not encoded as numeric. Only those two types are currently supported')
        else:
            zeros = np.zeros((len(self._ds_targets), self._n_classes), np.int32)
            # filling with 1s using numpy's fancy indexing:
            zeros[np.arange(len(self._ds_targets)), self._ds_targets] = 1
            self._ds_targets = zeros
            self._encoding = Encodings.one_hot

    def to_numeric_encoding(self):
        """
        Changes the target value encoding to numeric (0, 0, 0, 0, 1) -> 4
        Throws an exception if the problem is not a classification problem or it
        is not encoded with a suitable method to change to numeric.

        :return: None
        """

        if self._problem != Problems.classification:
            raise DatasetException(self, 'Trying to change encoding on a dataset which does not belong to a classification problem')
        elif self._encoding != Encodings.one_hot:
            raise DatasetException(self, 'Trying to apply numeric encoding to a dataset which is not encoded as one-hot. Only those two types are currently supported')
        else:
            self._ds_targets = np.argmax(self._ds_targets, axis=1)
            self._encoding = Encodings.numeric

    def normalise(self, method='standard'):
        """
        Normalizes the input values in the dataset

        :param: method (string in {'normal', 'linear', 'symmetric'}): desired
                normalisation method. Supported options:
                'standard': modifies each feature to an (mean 0, std 1) distribution
                'linear': modifies each feature linearly taking [min, max] to
                         [0, 1]
                'symmetric': modifies each feature linearly taking [min, max] to
                            [-1, 1]
        :return: None
        """
        if self._normalisation is not None:
            raise DatasetException(self, 'Dataset previously normalised')

        # preventing division-by-zero issues when normalising
        minv, maxv = np.min(self._ds_inputs, axis=0), np.max(self._ds_inputs, axis=0)
        equal_inds = [str(i) for i in range(len(minv)) if minv[i] == maxv[i]]
        if equal_inds:
            raise DatasetException(self, 'The following {} constant values, so no normalisation can be applied: {}. Please consider removing them'
                                   .format('feature has' if len(equal_inds) == 1 else 'features have', ', '.join(equal_inds)))

        if method == 'standard':
            std = np.std(self._ds_inputs, axis=0)
            mean = np.mean(self._ds_inputs, axis=0)
            self._ds_inputs = (self._ds_inputs - mean) / std
        elif method == 'linear':
            self._ds_inputs = (self._ds_inputs - minv) / (maxv - minv)
        elif method == 'symmetric':
            self._ds_inputs = (2 * self._ds_inputs - (maxv + minv)) / (maxv - minv)
        else:
            raise DatasetException(self, 'Invalid normalisation type')

        self._normalisation = method

    def split(self, training_prop, test_prop):
        """
        Splits the dataset input and target values.
        The validation proportion is inferred from the other two
        If the problem type is classification, attempt

        :param: training_prop (float): proportion of the dataset used for training.
        :param: test_prop (float): proportion of the dataset used for testing.
        :return: None
        """

        if training_prop + test_prop > 1:
            raise DatasetException(self, 'The training and testing proportions cannot add to more than 1')

        if self._problem == Problems.classification:
            # ensuring equal representation of classes in each dataset

            self._t_indices, self._te_indices, self._v_indices = [], [], []

            for inds in self.ind_in_class:

                n = len(inds)
                n_t = int(n * training_prop)
                n_te = int(n * test_prop)

                np.random.shuffle(inds)

                self._t_indices += inds[:n_t]
                self._te_indices += inds[n_t:n_t + n_te]
                self._v_indices += inds[n_t + n_te:]

            np.random.shuffle(self._t_indices)
            np.random.shuffle(self._te_indices)
            np.random.shuffle(self._v_indices)

        else:
            n = self._n_examples
            n_t = int(n * training_prop)
            n_te = int(n * test_prop)

            ind = np.random.permutation(n)

            self._t_indices = ind[:n_t]
            self._te_indices = ind[n_t:n_t + n_te]
            self._v_indices = ind[n_t + n_te:]

        self._to_see_next = 0


    def _subdata(self, inputs, targets, indices, transpose):
        """
        Internal method called by the methods that return subsets of the Dataset

        :param inputs (boolean): Whether to return the input values
               Default: True
        :param targets (boolean): Whether to return the target values
               Default: True
        :param indices (int): the indices of the subset within the whole dataset
        :param transpose (boolean): Transpose the returned arrays so that each
               example is a column instead of a row.
               Default: False
        :return: (numpy array) if one of the arguments is True
                 (numpy array, numpy array) if both arguments are True
                 None if both arguments are false
        """
        a_inputs, a_targets = self._ds_inputs[indices], self._ds_targets[indices]
        if transpose:
            a_inputs, a_targets = a_inputs.transpose(), a_targets.transpose()

        if inputs and targets:
            return a_inputs, a_targets
        elif inputs:
            return a_inputs
        elif targets:
            return a_targets
        # if both are false, return None

    def reset_minibatches(self, shuffle=False):
        """
        Sets up the minibatches to start over, optionally shuffling the training
        set.
        Raises an exception if the Dataset has not been split yet.

        :param: shuffle (boolean): Whether to shuffle the training set or not.
        :return: None
        """
        if self._to_see_next == -1:
            raise DatasetException(self, 'Attempting to get minibatch for a dataset which has not been split yet')

        if shuffle:
            np.random.shuffle(self._t_indices)

        self._to_see_next = 0

    # getters and setters
    # only the ones with sensible external use have been included. attributes
    # such as lists and dicts have been protected against external modification
    def get_n_features(self):
        """
        Returns the number of input features in the dataset

        :return: (int): The number of features in each example
        """
        return self._n_features

    def get_encoding(self):
        """
        Returns the encoding as a string

        :return: (string): The dataset target encoding.
        """
        return self._encoding.value

    def get_normalisation(self):
        """
        Returns the normalisation method

        :return: (string): The normalisation method applied to the input values
                 or None if the dataset hasn't been normalised yet.
        """
        return self._normalisation

    def get_data(self, inputs=True, targets=True, transpose=False):
        """
        Returns the inputs and/or targets in the dataset

        :param inputs (boolean): Whether to return the input values
               Default: True
        :param targets (boolean): Whether to return the target values
               Default: True
        :param transpose (boolean): Transpose the returned arrays so that each
               example is a column instead of a row.
               Default: False
        :return: (numpy array) if one of the arguments is True
                 (numpy array, numpy array) if both arguments are True
                 None if both arguments are false
        """
        return self._subdata(inputs, targets, np.arange(self._n_examples), transpose)
        
    def get_training_data(self, inputs=True, targets=True, transpose=False):
        """
        Returns the training inputs and/or targets in the dataset
        The arrays are empty if Dataset.split has not been called before

        :param inputs (boolean): Whether to return the input values
               Default: True
        :param targets (boolean): Whether to return the target values
               Default: True
        :param transpose (boolean): Transpose the returned arrays so that each
               example is a column instead of a row.
               Default: False
        :return: (numpy array) if one of the arguments is True
                 (numpy array, numpy array) if both arguments are True
                 None if both arguments are false
        """
        return self._subdata(inputs, targets, self._t_indices, transpose)

    def get_validation_data(self, inputs=True, targets=True, transpose=False):
        """
        Returns the validation inputs and/or targets in the dataset
        The arrays are empty if Dataset.split has not been called before

        :param inputs (boolean): Whether to return the input values
               Default: True
        :param targets (boolean): Whether to return the target values
               Default: True
        :param transpose (boolean): Transpose the returned arrays so that each
               example is a column instead of a row.
               Default: False
        :return: (numpy array) if one of the arguments is True
                 (numpy array, numpy array) if both arguments are True
                 None if both arguments are false
        """

        return self._subdata(inputs, targets, self._v_indices, transpose)

    def get_test_data(self, inputs=True, targets=True, transpose=False):
        """
        Returns the validation inputs and/or targets in the dataset
        The arrays are empty [] if Dataset.split has not been called before

        :param: inputs (boolean): Whether to return the input values
               Default: True
        :param: targets (boolean): Whether to return the target values
               Default: True
        :param transpose (boolean): Transpose the returned arrays so that each
               example is a column instead of a row.
               Default: False
        :return: (numpy array) if one of the arguments is True
                 (numpy array, numpy array) if both arguments are True
                 None if both arguments are false
        """

        return self._subdata(inputs, targets, self._te_indices, transpose)

    def get_minibatch(self, size, autoreset=False, transpose=False):
        """
        Returns the next minibaatch with the given size from the training set.
        If fewer unseen examples than 'size' are left, returns only those. If
        all examples have already been yielded, returns None, None (use
        reset_minibatches to start over) if autoreset is False and resets and
        yields the first minibatch otherwise.
        Raises an exception if the Dataset has not been split yet.

        :param size (int): The number of examples to return
        :param autoreset (boolean): Whether to start over and shuffle automatically
                                    when the last mini-batch is consumed
        :param transpose (boolean): Transpose the returned arrays so that each
               example is a column instead of a row.
               Default: False
        :return: (numpy array, numpy array) with the input and target values
                 respectively.
        """
        if self._to_see_next == -1:
            raise DatasetException(self, 'Attempting to get minibatch for a dataset which has not been split yet')

        n_training = len(self._t_indices)

        # if we have reached the last mini-batch earlier
        if self._to_see_next == n_training:
            if autoreset:
                np.random.shuffle(self._t_indices)
                self._to_see_next = 0
            else:
                return None, None

        from_x = self._to_see_next
        self._to_see_next = min(self._to_see_next + size, n_training)

        return self._subdata(True, True, self._t_indices[from_x: self._to_see_next], transpose)

    # getters for classification datasets
    def get_classes(self):
        """
        Returns the list of classes if the dataset is a classification one.
        Raises an exception if the problem is not classification.

        :return: (list): copy of the internal list of classes
        """

        if self._problem != Problems.classification:
            raise DatasetException(self,
                                   'Classes are only available for classification problem datasets')

        return list(self._classes)

    def get_class_mapping(self):
        """
        Returns the mapping between classes and indices. Raises an exception if
        the problem is not classification.

        :return: (dict): a copy of the internal dictionary that maps each class
                 name to its class number or index. None if the source file was
                 already coded with indices.
        """

        if self._problem != Problems.classification:
            raise DatasetException(self,
                                   'Class mappings are only available for classification problem datasets')

        return None if self._mapping_dict is None else dict(self._mapping_dict)

    def get_n_classes(self):
        """
        Returns the number of classes in the dataset. Raises an exception if
        the problem is not classification.

        :return: (dict): a copy of the internal dictionary that maps each class
                 name to its class number or index. None if the source file was
                 already coded with indices.
        """

        if self._problem != Problems.classification:
            raise DatasetException(self, 'Class mappings are only available for classification problem datasets')

        return len(self._classes)

    # representation methods
    def get_information(self):
        """
        Returns a description of the dataset parameters as a string

        :return: (str): The dataset information
        """

        # using python 3.5 recommended string formatting
        n = 'Dataset: {}'.format(self.name)
        p = 'Problem: {}'.format(self._problem.value)
        e = 'Encoding: {}'.format(self._encoding.value)
        nr = 'Normalisation: {}'.format(self._normalisation)
        ne = '{} examples'.format(len(self._ds_inputs))
        c = '{} classes: {}'.format(self._n_classes, ', '. join(self._classes)) if self._problem == Problems.classification else ''

        return '\n'.join([n, p, e, nr, ne, c])
