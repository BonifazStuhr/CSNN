from abc import ABCMeta, abstractmethod

class ITrainer(metaclass=ABCMeta):
    """
    The ITrainer provides the interface for trainer classes, such as a multi gpu backprob trainer.
    """
    @abstractmethod
    def createTraining(self, model, input_data, labels):
        """
        Interface Methode: Creates the training of the model.
        :param model: (tf_graph_tensor) The model to train.
        :param input_data: (tf_graph_tensor) The data to train the model.
        :param labels: (tf_graph_tensor) The corresponding labels to the data.
        :return: training_ops: (tf_graph_tensors) The operations to exexute the training in a tf.session.
        """
        raise NotImplementedError('Not implemented')

    @abstractmethod
    def createValidation(self, model, input_data, labels):
        """
        Interface Methode: Creates the validation of the trained model.
        :param model: (tf_graph_tensor) The model to validate.
        :param input_data: (tf_graph_tensor) The data to validate the model.
        :param labels: (tf_graph_tensor) The corresponding labels to the data.
        :return: validation_ops: (tf_graph_tensors) The operations to exexute the training in a tf.session.
        """
        raise NotImplementedError('Not implemented')



