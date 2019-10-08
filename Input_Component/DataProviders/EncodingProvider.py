import os
import numpy as np
import tensorflow as tf

from Input_Component.ADataProvider import ADataProvider

class EncodingProvider(ADataProvider):
    """
    The EncodingProvider reads the encoded dataset from various locations and provides the dataset in various forms.
    The EncodingProvider is not responsible for augmenting the dataset!

    :Attributes:
        __num_classes:     (integer) The number of classes of the dataset.
    """

    def __init__(self, encoding_name, data_encoding_tfrecord_shape, label_encoding_tfrecord_shape, train_size,
                 eval_size, test_size, processable_at_once=True, prepreporcessing_type="'None'"):
        """
        Constructor, initialize member variables.
        :param encoding_name: (String) The name of the encoded dataset.
        :param data_encoding_tfrecord_shape: (Array) The shape of the data in the data tfrecord to parse.
        :param label_encoding_tfrecord_shape: (Array) The shape of the label in the label tfrecord to parse.
        :param train_size: (Integer) The size of the training set of the encoding.
        :param eval_size: (Integer) The size of the eval set of the encoding.
        :param test_size: (Integer) The size of the test set of the encoding.
        :param dataset_processable_at_once: (Boolean) Is the dataset processable at once?
        :param prepreprocessing_type: (String) The type of the prepreprocessing before getting the dataset or writing
                                      the tfrecord.
        """
        self.__num_classes = label_encoding_tfrecord_shape[0]

        super().__init__(dataset_path=os.path.join('data', encoding_name),
                         dataset_name=encoding_name,
                         dataset_size=train_size+eval_size+test_size,
                         train_size=train_size,
                         eval_size=eval_size,
                         test_size=test_size,
                         prepreporcessing_type=prepreporcessing_type,
                         dataset_processable_at_once=processable_at_once,
                         tfrecord_shapes={"data":data_encoding_tfrecord_shape, "label":label_encoding_tfrecord_shape},
                         tfrecord_datatyps={"data":"float32", "label":"uint8"})

    def loadDataset(self):
        """
        Method: Reads and returns the dataset.
        :return: x_train: (Array) The train data.
        :return: y_train: (Array) The train label.
        :return: x_eval: (Array) The eval data.
        :return: y_eval: (Array) The eval label.
        :return: x_test: (Array) The test data.
        :return: y_test: (Array) The test label.
        """
        # No jet implemented. Implement if needed.
        raise NotImplementedError('Not implemented')

    def getSplittedDatasetInNumpy(self, depth_first=False, onehot=True, random_seed=None):
        """
        Reads and returns the data of Cifar10 in numpy format with the set split (setDatasetSplit).
        :param depth_first: (Boolean) If true the image dimensions are NCHW. False by default.
        :param onehot: (Boolean) If true the label is converted to onehot encoding. True by default.
        :return: dataset: (Dictionary) The dataset e.g. {"x_train":(train_size, 3, 32, 32), "y_train":(train_size,) or
                           if onehot (train_size, 10), x_eval....
        """
        # No jet implemented. Implement if needed.
        raise NotImplementedError('Not implemented')

    def getNumReadInBatches(self, mode):
        """
        Interface Method: Returns the number of batches for a given mode.
        :param mode: (String) The mode of the saved record.
        :return: num_batches: (Integer) The number of batches for the given mode.
        """
        # No jet implemented. Implement if needed.
        raise NotImplementedError('Not implemented for EncodingProvider')

    def getNextReadInBatchInNumpy(self, mode):
        """
        Interface Method: Returns the next batch for the given mode.
        :param mode: (String) The mode of the saved record.
        :return: batch: (Array of Dictionaries) The next batch of the dataset for the given mode in the form
        e.g.: {"data":(read_in_batch_size, 1, 28, 28), "label":(read_in_batch_size, ) or if onehot (read_in_batch_size, 10).
        """
        # No jet implemented. Implement if needed.
        raise NotImplementedError('Not implemented for EncodingProvider')

    def getXShot(self, mode, x, sess, batch_size, tfrecord_inputpipeline_config):
        """
        Returns x samples for each class for the given mode
        :param mode: (String) The mode of the saved record.
        :param x: (Integer) The number of samples per class.
        :param depth_first: (Boolean) If true the image dimensions are NCHW. False by default.
        :param onehot: (Boolean) If true the label is converted to onehot encoding. False by default.
        :return: sample_data: (Dictionary) E.g.: The sample_data with the form {"data":(x, 3, 32, 32), "label":(x,) or
                              if onehot (x, 10).
        """

        tfrecord_inputpipeline_config["shuffleMB"] = 0

        iterator_val, file_placeholder_val = \
            self.getTfRecordInputPipelineIteratorsAndPlaceholdersFor("val", batch_size,
                                                                     tfrecord_inputpipeline_config["val"])

        # Get the names of the input files to feed the pipeline later with train/eval/test files.
        data_files = self.getTfRecordFileNames()

        sess.run((iterator_val.initializer), feed_dict={file_placeholder_val: data_files[mode]})

        next_element_op, label_next_element_op = iterator_val.get_next()

        labels, data = sess.run((label_next_element_op, next_element_op))

        try:
            while True:
                batch_label, batch_data = sess.run((label_next_element_op, next_element_op))

                data = np.concatenate((data, batch_data))
                labels = np.concatenate((labels, batch_label))

        except tf.errors.OutOfRangeError:
            return self._extractXClassSamples(data, np.argmax(labels, axis=1), x, self.__num_classes)




