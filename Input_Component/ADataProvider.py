import os
import sys
import tensorflow as tf
import numpy as np
from abc import ABCMeta, abstractmethod

from Input_Component.TfRecord.TfRecordProvider import TfRecordProvider

class ADataProvider(metaclass=ABCMeta):
    """
    The ADataProvider reads the dataset from tf records and provides the dataset in various forms.
    The ADataProvider is not responsible for augmenting the dataset!

    :Attributes:
        __dataset_path:                    (String) The path to the gz files of the mnist dataset.
        __dataset_name:                    (String) The name of the dataset.
        __dataset_size:                    (Integer) The size of the dataset (for mnist 70000).
        __train_size:                      (Integer) The size of the training set. 60000 by default.
        __eval_size:                       (Integer) The size of the eval set. 0 by default.
        __test_size:                       (Integer) The size of the test set. 10000 by default.
        __tfRecordProvider:                (TfRecordProvider) The class to construct the tfrecord pipeline.
        __prepreprocessing_type:           (String) The type of the prepreprocessing before getting the dataset or
                                           writing the tfrecord.
        __dataset_processable_at_once:     (Boolean) Is it possible to load and process the whole dataset in ram?
        __tfrecord_shapes:                 (Dictionary) The shapes of a entry in the tfrecord file
                                            e.g. {"data":[1,28,28], "label":[10]}.
        __tfrecord_datatyps:               (Dictionary) The datatyps of a entry in the tfrecord
                                            e.g. {"data":"float32", "label":"uint8"}.
        __tf_datatyps_after_parse:         (Dictionary) The datatyps of a entry after it is parsed
                                            e.g. {"data":"tf.float32", "label":"uint8"}.
        __num_classes:                     (Integer) The number of classes of the dataset.
        __read_in_size:                    (Integer) The size of the dataset to read in.
        __read_in_shape:                   (Array) The shape of the input date to read in
        __read_in_images:                  (Array) If the dataset is read in in numpy and fits in memory the input data
                                            will be saved.
        __read_in_labels:                  (Array) If the dataset is read in in numpy and fits in memory the label data
                                            will be saved.
        __read_in_dataset:                 (Boolean) If false the dateset will be read from disk. If true the datset is
                                            already in memory.
    """

    def __init__(self, dataset_path, dataset_name, dataset_size, train_size, eval_size, test_size, prepreporcessing_type,
                 dataset_processable_at_once, tfrecord_shapes, tfrecord_datatyps, num_classes=None, read_in_size=None,
                 read_in_shape=None, tf_datatyps_after_parse={"data": tf.float32, "label": tf.uint8}):
        """
        Constructor, initialize member variables.
        :param dataset_path: (String) The path to the gz files of the mnist dataset.
        :param dataset_name: (String) The name of the dataset.
        :param dataset_size: (Integer) The size of the dataset (for mnist 70000).
        :param train_size: (Integer) The size of the training set.
        :param eval_size: (Integer) The size of the eval set.
        :param test_size: (Integer) The size of the test set.
        :param prepreprocessing_type: (String) The type of the prepreprocessing before getting the dataset or writing
                                      the tfrecord.
        :param dataset_processable_at_once: (Boolean) Is the dataset processable at once?
        :param tfrecord_shapes: (Dictionary) The shapes of a entry in the tfrecord file
                                 e.g. {"data":[1,28,28], "label":[10]}.
        :param tfrecord_datatyps: (Dictionary) The datatyps of a entry in the tfrecord
                                    e.g. {"data":"float32", "label":"uint8"}.
        :param num_classes: (Integer) The number of classes of the dataset.
        :param read_in_size: (Integer) The size of the dataset to read in.
        :param read_in_shape: (Array) The shape of the input date to read in
        :param tf_datatyps_after_parse: (Dictionary) The datatyps of a entry after it is parsed
                                         e.g. {"data":"tf.float32", "label":"uint8"}.
        """
        self.__dataset_path = dataset_path
        self.__dataset_name = dataset_name

        self.__dataset_size = dataset_size

        # Set values for the dataset split.
        self.__train_size = train_size
        self.__eval_size = eval_size
        self.__test_size = test_size

        self.__prepreprocessing_type = prepreporcessing_type

        self.__dataset_processable_at_once = dataset_processable_at_once

        self.__tfrecord_shapes = tfrecord_shapes
        self.__tfrecord_datatyps = tfrecord_datatyps
        self.__tf_datatyps_after_parse = tf_datatyps_after_parse

        # Check if the split is possible.
        assert train_size + eval_size + test_size <= self.__dataset_size

        self.__tfRecordProvider = TfRecordProvider()

        self.__num_classes = num_classes
        self.__read_in_size = read_in_size
        self.__read_in_shape = read_in_shape
        self.__read_in_images = None
        self.__read_in_labels = None
        self.__read_in_dataset = False

    def setDatasetSplit(self, dataset_spilt):
        """
        Sets the split of the dataset.
        :param dataset_spilt: (Array) The array containing the split numbers in the order: [train_num, eval_num,
        test_num, "prepreprocessing_type"].
        """
        self.__train_size = dataset_spilt[0]
        self.__eval_size = dataset_spilt[1]
        self.__test_size = dataset_spilt[2]
        self.__dataset_size = dataset_spilt[0]+dataset_spilt[1] + dataset_spilt[2]

        self.__prepreprocessing_type = dataset_spilt[3]

    def getTfRecordInputPipelineIteratorsAndPlaceholdersFor(self, mode, batch_size, input_pipeline_config):
        """
        Returns the input pipeline for the tfrecord and the given mode, batch_size and input pipeline configuration.
        :param mode: (String) The mode of the saved record.
        :param batch_size: (Integer) The desired batch size.
        :param input_pipeline_config: (Dictionary) The configuration of the dataset input pipeline.
        :return: dataset_iterator: (tf.data.Iterator)  The dataset iterator with the desired batch size.
        :return: dataset_file_placeholder: (tf.placeholder) The placeholder for the dataset file to parse.
        """
        with tf.name_scope(str(self.__dataset_name) +'Data_'+mode), tf.device('/cpu:0'):
            if mode is "train":
                dataset_input, dataset_file_placeholder = self.__tfRecordProvider.getInputPipelineFrom(
                    input_pipeline_config, batch_size, self.__tfrecord_shapes, self.__tfrecord_datatyps,
                    self.__tf_datatyps_after_parse)
            else:
                dataset_input, dataset_file_placeholder = self.__tfRecordProvider.getInputPipelineFrom(
                    input_pipeline_config, batch_size, self.__tfrecord_shapes, self.__tfrecord_datatyps,
                    self.__tf_datatyps_after_parse, False)

            dataset_iterator = dataset_input.make_initializable_iterator()

            return dataset_iterator, dataset_file_placeholder

    def getTfRecordFileNames(self):
        """
        Returns the filepaths containing of the tf records.
        :return: dataset_files: (Dictionary) The train, test, eval file names of the dataset.
        """
        dataset_spilt = "[" + str(self.__train_size) + ", " + str(self.__eval_size) + ", " + str(
            self.__test_size) + ", " + self.__prepreprocessing_type + "]" + "_"

        file_path = os.path.dirname(sys.modules['__main__'].__file__) + "/" + str(self.__dataset_path + "/"
                                                                                  + self.__dataset_name + "_"
                                                                                  + dataset_spilt)

        # TODO: Load all files if more exist
        dataset_files = {"train": [file_path + "train" + ".tfrecords"],
                        "eval": [file_path + "eval" + ".tfrecords"],
                        "test": [file_path + "test" + ".tfrecords"]}

        return dataset_files

    def getSplittedDatasetInNumpy(self, depth_first=False, onehot=True, random_seed=None):
        """
        Reads and returns the data of Cifar10 in numpy format with the set split (setDatasetSplit).
        :param depth_first: (Boolean) If true the image dimensions are NCHW. False by default.
        :param onehot: (Boolean) If true the label is converted to onehot encoding. True by default.
        :return: dataset: (Dictionary) The dataset e.g. {"x_train":(train_size, 3, 32, 32), "y_train":(train_size,)
                          or if onehot (train_size, 10), x_eval....
        """
        if not self.__read_in_dataset:
            (x_train, y_train), (x_eval, y_eval), (x_test, y_test) = self.loadDataset()

            self.__read_in_images, self.__read_in_labels = self.convertDatasetToNumpy(x_train, y_train, x_eval, y_eval,
                                                            x_test, y_test, self.__read_in_shape, self.__read_in_size,
                                                            self.__num_classes, depth_first, onehot)
            self.__read_in_dataset=True

        dataset = {
            "x_train": self.__read_in_images[:self.__train_size],
            "y_train": self.__read_in_labels[:self.__train_size],

            "x_eval": self.__read_in_images[self.__train_size:self.__train_size + self.__eval_size],
            "y_eval": self.__read_in_labels[self.__train_size:self.__train_size + self.__eval_size],

            "x_test": self.__read_in_images[-self.__test_size:],
            "y_test": self.__read_in_labels[-self.__test_size:],
        }

        return dataset

    def getXShotInNumpy(self, mode, x, depth_first=False, onehot=False):
        """
        Returns x samples for each class for the given mode
        :param mode: (String) The mode of the saved record.
        :param x: (Integer) The number of samples per class.
        :param depth_first: (Boolean) If true the image dimensions are NCHW. False by default.
        :param onehot: (Boolean) If true the label is converted to onehot encoding. False by default.
        :return: sample_data: (Dictionary) The sample_data e.g. {"data":(x, 1, 28, 28), "label":(x,) or if onehot (x, 10).
        """
        if not self.__read_in_images:
            if not self.__read_in_images:
                (x_train, y_train), (x_eval, y_eval), (x_test, y_test) = self.loadDataset()
                self.__read_in_images, self.__read_in_labels = self.convertDatasetToNumpy(x_train, y_train, x_eval,
                                                                                          y_eval,
                                                                                          x_test, y_test,
                                                                                          self.__read_in_shape,
                                                                                          self.__read_in_size,
                                                                                          self.__num_classes,
                                                                                          depth_first, onehot)


        if mode is "train":
            return self._extractXClassSamples(self.__read_in_images[:self.__train_size],
                                              self.__read_in_labels[:self.__train_size], x, self.__num_classes)
        elif mode is "eval":
            return self._extractXClassSamples(
                self.__read_in_images[self.__train_size:self.__train_size + self.__eval_size],
                self.__read_in_labels[self.__train_size:self.__train_size + self.__eval_size], x, self.__num_classes)
        elif mode is "test":
            return self._extractXClassSamples(
             self.__read_in_images[-self.__test_size:], self.__read_in_labels[-self.__test_size:], x, self.__num_classes)

    def _extractXClassSamples(self, data, labels, x, num_classes):
        """
        Reads, converts and returns the dataset_part of Cifar10 in numpy format.
        :param data: (Array) The data from which to extract the samples.
        :param labels: (Array) The labels from which to extract the samples.
        :param x: (Integer) The number of samples to extract.
        :param num_classes: (Integer) The number of classes in the dataset.
        :return: labels: (np.array) The labels in (datasetpart_size,) shape.
        """
        samples = []
        samples_label = []
        for i in range(0, num_classes):
            samples.append(data[i == labels][0:x])
            samples_label.append(np.full((x), i))
        return {"data": samples, "label": samples_label}

    def convertDatasetToNumpy(self, x_train, y_train, x_eval, y_eval, x_test, y_test, shape, dataset_size, num_classes,
                              depth_first, onehot):
        """
        Reads, converts and returns the dataset_part of Cifar10 in numpy format.
        :param x_train: (Array) The train data.
        :param y_train: (Array) The train label.
        :param x_eval: (Array) The eval data.
        :param y_eval: (Array) The eval label.
        :param x_test: (Array) The test data.
        :param y_test: (Array) The test label.
        :param shape: (Array) The shape of the in input data.
        :param dataset_size: (Integer) The size of the dataset.
        :param depth_first: (Boolean) If true the image dimensions are NCHW. False by default.
        :param onehot: (Boolean) If true the label is converted to onehot encoding. True by default.
        :return: images: (np.array) The images in (datasetpart_size, 3, 32, 32) shape.
        :return: labels: (np.array) The labels in (datasetpart_size,) shape.
        """
        if (x_eval is None) and (x_test is None):
            input = x_train
            labels = y_train
        elif x_eval is None:
            input = np.concatenate((x_train, x_test), axis=0)
            labels = np.concatenate((y_train, y_test), axis=0)
        else:
            input = np.concatenate((x_train, x_eval, x_test), axis=0)
            labels = np.concatenate((y_train, y_eval, y_test), axis=0)

        shape = np.insert(shape, 0, -1, axis=0)
        input = np.reshape(input, shape)
        input = input.astype('float32')
        labels = np.reshape(labels, [-1, ])
        labels = labels.astype(np.uint8)

        shape[0] = dataset_size
        assert list(input.shape) == list(shape) and input.dtype == np.float32
        assert labels.shape == (dataset_size, ) and labels.dtype == np.uint8
        assert np.min(input) == 0 and np.max(input) == 255
        assert np.min(labels) == 0 and np.max(labels) == (num_classes - 1)

        if depth_first:
            input = np.transpose(input, [0, 3, 1, 2])

        # If onehot encoding is needed, the labels will be encoded.
        if onehot:
            onehot_labels = np.zeros((labels.size, np.max(labels) + 1), dtype=np.float32)
            onehot_labels[np.arange(labels.size), labels] = 1.0
            labels = np.array(onehot_labels, dtype=np.uint8)
            assert labels.shape == (dataset_size, num_classes) and labels.dtype == np.uint8

        return input, labels

    def datasetProcessableAtOnce(self):
        """
        Checks if the size of the dataset is small enough to process it at once.
        :return: processable_at_once: (Boolean) If true, the dataset is processable at once.
        """
        return self.__dataset_processable_at_once

    @abstractmethod
    def loadDataset(self):
        """
        Interface Method: Reads and returns the dataset.
        :return: x_train: (Array) The train data.
        :return: y_train: (Array) The train label.
        :return: x_eval: (Array) The eval data.
        :return: y_eval: (Array) TThe eval label.
        :return: x_test: (Array) The test data.
        :return: y_test: (Array) The test label.
        """
        raise NotImplementedError('Not implemented')

    @abstractmethod
    def getNumReadInBatches(self, mode):
        """
        Interface Method: Returns the number of batches for a given mode.
        :param mode: (String) The mode of the saved record.
        :return: num_batches: (Integer) : The number of batches for the given mode.
        """
        raise NotImplementedError('Not implemented')

    @abstractmethod
    def getNextReadInBatchInNumpy(self, mode):
        """
        Interface Method: Returns the next batch for the given mode.
        :param mode: (String) The mode of the saved record.
        :return: batch: (Array of Dictionaries) The next batch of the dataset for the given mode in the form e.g.:
        {"data":(read_in_batch_size, 1, 28, 28), "label":(read_in_batch_size, ) or if onehot (read_in_batch_size, 10).
        """
        raise NotImplementedError('Not implemented')

    def _getDatasetPath(self):
        """
        Returns the path to the "raw" data of the dataset.
        :return: dataset_path: (String) The path to the dataset.
        """
        return self.__dataset_path

    def _getTrainSize(self):
        """
        Returns the size of the train dataset.
        :return: size: (Integer) The size of the train dataset.
        """
        return self.__train_size

    def _getEvalSize(self):
        """
        Returns the size of the eval dataset.
        :return: size: (Integer) The size of the eval dataset.
        """
        return self.__eval_size

    def _getTestSize(self):
        """
        Returns the size of the test dataset.
        :return: size: (Integer) The size of the test dataset.
        """
        return self.__test_size

    def getSetSize(self, mode):
        """
        Returns the size of the dataset for the given mode.
        :param mode: (String) The mode of the saved record.
        :return: size: (Integer) The size of the dataset.
        """
        if mode is "train":
            return self.__train_size
        elif mode is "eval":
            return self.__eval_size
        elif mode is "test":
            return self.__test_size
        return None

    def getEntryShape(self, entry_name):
        """
        Returns the size of the test dataset.
        :param: entry_name: (String) The name of the entry.
        :return: shape: (Array) The shape of the entry.
        """
        return self.__tfrecord_shapes[entry_name]

    def getEntryDatatyp(self, entry_name):
        """
        Returns the size of the test dataset.
        :param: entry_name: (String) The name of the entry.
        :return: type: (tf.datatyp) The datatyp of the entry.
        """
        return self.__tfrecord_datatyps[entry_name]