import tensorflow as tf

from Preprocessing_Component.IPreprocessing import IPreprocessing

class MaxDivNormalizer(IPreprocessing):
    """
    The MaxDivNormalizer normalizes by div. with trough the may value

    :Attributes:
        __max_value:    (Integer) The max_value of the dataset.
    """
    def __init__(self, max_value):
        """
        Constructor, initialize member variables.
        :param max_value : The max_value of the dataset.
        """
        self.__max_value = max_value

    def preprocessingFn(self, input):
        """
        The function (or graph part) of the preprocessing.
        This function normalizes the input via the max_value.
        :param input: (tf_graph_tensor) The input to normalize.
        :returns normalized_input: (tf_graph_tensor) The input to normalize.
        """
        with tf.name_scope("NormalizeTrough" + str(self.__max_value)):
            return tf.truediv(input, float(self.__max_value))

    def preprocessingInNumpy(self, dataset):
        """
        This function normalizes the input via the max_value.
        :param input: (Dictionary) The dataset e.g. {"x_train":(train_size, 3, 32, 32), "y_train":(train_size,) or if
                        onehot (train_size, 10), x_eval....
        :returns normalized_dataset: (Dictionary) The dataset e.g. {"x_train":(train_size, 3, 32, 32),
                                      "y_train":(train_size,) or if onehot (train_size, 10), x_eval....
        """
        dataset["x_train"] = dataset["x_train"] / float(self.__max_value)
        if dataset["x_eval"] is not None:
            dataset["x_eval"] = dataset["x_eval"] / float(self.__max_value)
        if dataset["x_test"] is not None:
            dataset["x_test"] = dataset["x_test"] / float(self.__max_value)

        return dataset