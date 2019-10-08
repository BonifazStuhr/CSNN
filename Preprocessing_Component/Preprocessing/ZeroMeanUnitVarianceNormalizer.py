import tensorflow as tf
import numpy as np

from Preprocessing_Component.IPreprocessing import IPreprocessing

class ZeroMeanUnitVarianceNormalizer(IPreprocessing):
    """
    The ZeroMeanUnitVariance normalizes the input for zero mean and unit variance via the given mean and standard
    deviation values of the dataset.

    Mean subtraction is the most common form of preprocessing.
    It involves subtracting the mean across every individual feature in the data, and has the geometric
    interpretation of centering the cloud of data around the origin along every dimension.
    In numpy, this operation would be implemented as: X -= np.mean(X, axis = 0). With images specifically,
    for convenience it can be common to subtract a single value from all pixels (e.g. X -= np.mean(X)),
    or to do so separately across the three color channels.

    Normalization refers to normalizing the data dimensions so that they are of approximately the same scale.
    There are two common ways of achieving this normalization. One is to divide each dimension by its standard
    deviation, once it has been zero-centered: (X /= np.std(X, axis = 0)). Another form of this preprocessing
    normalizes each dimension so that the min and max along the dimension is -1 and 1 respectively.
    It only makes sense to apply this preprocessing if you have a reason to believe that different
    input features have different scales (or units), but they should be of approximately equal importance to the
    learning algorithm. In case of images, the relative scales of pixels are already approximately equal
    (and in range from 0 to 255), so it is not strictly necessary to perform this additional preprocessing step.

    :Attributes:
        __dataset_mean: (Array) The mean values of the dataset for each channel.
        __dataset_std:  (Array) The standard deviation values of the dataset for each channel.
    """
    def __init__(self, dataset_mean=None, dataset_std=None):
        """
        Constructor, initialize member variables.
        :param dataset_mean: (Array) The mean values of the dataset. Default None.
        :param dataset_variance: (Array) The standard deviation values of the dataset. Default None.
        """
        self.__dataset_mean = dataset_mean
        self.__dataset_std = dataset_std

    def preprocessingFn(self, input):
        """
        The function (or graph part) of the preprocessing.
        This function normalizes the input via the mean and standard deviation for zero mean and unit variance.
        :param input: (tf_graph_tensor) The input to normalize.
        :returns normalized_input: (tf_graph_tensor) The input to normalize.
        """
        with tf.name_scope("ZeroMeanUnitVariance"):
            import numpy as np
            input = tf.subtract(input, np.expand_dims(np.expand_dims(np.expand_dims(self.__dataset_mean, axis=0),
                                                                     axis=2), axis=3).astype(np.float32))
            # This function forces Python 3 division operator semantics
            # where all integer arguments are cast to floating types first.
            return tf.truediv(input, np.expand_dims(np.expand_dims(np.expand_dims(self.__dataset_std, axis=0),
                                                                   axis=2), axis=3).astype(np.float32))


    def preprocessingInNumpy(self, dataset):
        """
        This function normalizes the input via the mean and standard deviation for zero mean and unit variance.
        :param input: (Dictionary) The dataset e.g. {"x_train":(train_size, 3, 32, 32), "y_train":(train_size,) or if
                        onehot (train_size, 10), x_eval....
        :returns normalized_dataset: (Dictionary) The dataset e.g. {"x_train":(train_size, 3, 32, 32),
                                      "y_train":(train_size,) or if onehot (train_size, 10), x_eval....
        """
        mean = np.mean(dataset["x_train"], axis=(0, 1, 2, 3))
        std = np.std(dataset["x_train"], axis=(0, 1, 2, 3))
        dataset["x_train"] = (dataset["x_train"] - mean) / (std + 1e-7)
        if dataset["x_eval"] is not None:
            dataset["x_eval"] = (dataset["x_eval"] - mean) / (std + 1e-7)
        if dataset["x_test"] is not None:
            dataset["x_test"] = (dataset["x_test"] - mean) / (std + 1e-7)
        return dataset
