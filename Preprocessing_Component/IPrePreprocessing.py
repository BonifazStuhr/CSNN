from abc import ABCMeta, abstractmethod

class IPrePreprocessing(metaclass=ABCMeta):
    """
    The IPrePreprocessing provides the interface for prepreprocessing classes.
    """
    @abstractmethod
    def process(self, train_data, eval_data, test_data):
        """
        Interface Methode: The function of the prepreprocessing.
        This function is the place to implement the prepreprocessing logic.
        :param train_data: (Dictionaries) The train input to prepreprocess.
        :param eval_data: (Dictionaries) The eval input to prepreprocess.
        :param test_data: (Dictionaries) The test input to prepreprocess.
        :returns preprocessed_input: The prepreprocessed input.
        """
        raise NotImplementedError('Not implemented')



