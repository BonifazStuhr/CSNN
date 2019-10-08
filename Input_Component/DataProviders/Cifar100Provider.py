import os
from keras.datasets import cifar100

from Input_Component.ADataProvider import ADataProvider

class Cifar100Provider(ADataProvider):
    """
    The Cifar100Provider reads the Cifar100 dataset and provides the dataset in various forms.
    The Cifar100Provider is not responsible for augmenting the dataset!
    """

    def __init__(self, train_size=50000, eval_size=0, test_size=10000, prepreporcessing_type=""'None'"",
                 data_shape=[3,32,32]):
        """
        Constructor, initialize member variables.
        :param train_size: (Integer) The size of the training set. 50000 by default.
        :param eval_size: (Integer) The size of the eval set. 0 by default.
        :param test_size: (Integer) The size of the test set. 10000 by default.
        :param prepreprocessing_type: (String) The type of the prepreprocessing before getting the dataset or writing
                                      the tfrecord. "'None'" by default.
        :param data_shape: (Array) The shape of a data_entry for the tfrecord. [3,32,32] by default.
        """
        super().__init__(dataset_path=os.path.join('data', 'Cifar100'),
                         dataset_name='Cifar100',
                         dataset_size=train_size+eval_size+test_size,
                         train_size=train_size,
                         eval_size=eval_size,
                         test_size=test_size,
                         prepreporcessing_type=prepreporcessing_type,
                         dataset_processable_at_once=True,
                         num_classes=100,
                         read_in_size=60000,
                         read_in_shape=[32, 32, 3],
                         tfrecord_shapes={"data":data_shape, "label":[100]},
                         tfrecord_datatyps={"data":"uint8", "label":"uint8"})

    def loadDataset(self):
        """
        Reads and returns the dataset.
        :return: x_train: (Array) The train data.
        :return: y_train: (Array) The train label.
        :return: x_eval: (Array) The eval data.
        :return: y_eval: (Array) The eval label.
        :return: x_test: (Array) The test data.
        :return: y_test: (Array) The test label.
        """
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        return (x_train, y_train), (None, None), (x_test, y_test)

    def getNumReadInBatches(self, mode):
        """
        Interface Method: Returns the number of batches for a given mode.
        :param mode: (String) The mode of the saved record.
        :return: num_batches: (Integer) The number of batches for the given mode.
        """
        # No need to implement this function for Cifar100, because Cifar100 can be read at once.
        raise NotImplementedError('Not implemented for Cifar100Provider')

    def getNextReadInBatchInNumpy(self, mode):
        """
        Interface Method: Returns the next batch for the given mode.
        :param mode: (String) The mode of the saved record.
        :return: batch: (Array of Dictionaries) The next batch of the dataset for the given mode in the form
        e.g.: {"data":(read_in_batch_size, 3, 32, 32), "label":(read_in_batch_size, )
        or if onehot (read_in_batch_size, 100).
        """
        # No need to implement this function for Cifar100, because Cifar100 can be read at once.
        raise NotImplementedError('Not implemented for Cifar100Provider')


