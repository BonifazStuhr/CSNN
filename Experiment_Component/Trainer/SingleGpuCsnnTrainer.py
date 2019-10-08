import tensorflow as tf

from Experiment_Component.ITrainer import ITrainer

class SingleGpuCsnnTrainer(ITrainer):
    """
    The SingleGpuCsnnTrainer trains a model via the csnn learning algorithm.
    It handels both, the mask and the convolution weight training.
    """
    def createValidation(self, model, input_data, labels):
        """
        Creates the validation of the model.
        :param model: (tf_graph_tensor) The model to validate.
        :param input_data: (tf_graph_tensor) The data to validate the model.
        :param labels: (tf_graph_tensor) The corresponding labels to the data.
        :return: distances_per_layer: (tf_graph_tensors) The representation of each layer. The last layers
                                        representation is the csnn representation used for tasks like classification
        :return: patches_out: (tf_graph_tensors) Returns the image_patches of each layer for visualization.
        :return: labels: (tf_graph_tensors) If the data is labeled, the labels for the distances a
                            nd patches will be returned.
        """
        with tf.variable_scope(tf.get_variable_scope()) as outer_scope:
            with tf.name_scope('Validation'):
                # Reuse the current variables of the model.
                outer_scope.reuse_variables()
                distances_per_layer, patches_out = model.getValOp(input_data)
                return [distances_per_layer], [patches_out], [labels]

    def createTraining(self, model, input_data, labels):
        """
        Creates the validation of the model.
        :param model: (tf_graph_tensor) The model to validate.
        :param input_data: (tf_graph_tensor) The data to validate the model.
        :param labels: (tf_graph_tensor) The corresponding labels to the data.
        :return: train_ops: (tf_graph_tensors) The operations to execute the training in a tf.session.
        :return: apply_learning_weights: (tf_graph_tensors) The operations to apply the delta weights.
        :return: increment_global_step_op: (tf_graph_tensors) The operations to increment the global step
        """
        with tf.variable_scope(tf.get_variable_scope()):
            with tf.name_scope('Training'):
                train_op = model.getTrainOp(input_data)
                global_step = tf.train.get_or_create_global_step()
                apply_learning_weights, increment_global_step_op = model.applySomLearning(global_step, 1)
                return train_op, apply_learning_weights, increment_global_step_op


