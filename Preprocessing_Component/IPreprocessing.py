from abc import ABCMeta, abstractmethod

class IPreprocessing(metaclass=ABCMeta):
    """
    The IPreprocessing provides the interface for preprocessing classes in tf.
    """
    @abstractmethod
    def preprocessingFn(self, input):
        """
        Interface Methode: The function (or graph part) of the preprocessing.
        This function is the place to implement the preprocessing logic.
        :param input: (tf_graph_tensor) The input to preprocess.
        :returns normalized_input: (tf_graph_tensor)  The preprocessed input.
        """
        raise NotImplementedError('Not implemented')



