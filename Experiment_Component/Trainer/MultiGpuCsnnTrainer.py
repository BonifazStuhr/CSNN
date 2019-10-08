import tensorflow as tf

from Experiment_Component.ITrainer import ITrainer

class MultiGpuCsnnTrainer(ITrainer):
    """
    The MultiGpuCsnnTrainer trains a model via the csnn learning algorithm.
    It handels both, the mask and the convolution weight training.

     :Attributes:
        __num_gpus :    (Integer) The number of gpus to use.
        __controller :  (String) The name of the controller. The controller is the CPU/GPU where the weight updates are
                                 accumulated and applied to the model.
    """

    def __init__(self, num_gpus, controller="/cpu:0"):
        """
        Constructor, initialize member variables.
        :param num_gpus: (Integer) The number of gpus to use.
        :param controller: (String) The name of the controller. The controller is the CPU/GPU where the weight updates are
                           accumulated and applied to the model.
        """
        self.__num_gpus = num_gpus
        self.__controller = controller

    def createValidation(self, model, input_data, labels):
        """
        Creates the validation of the model.
        :param model: (tf_graph_tensor) The model to validate.
        :param input_data: (tf_graph_tensor) The data to validate the model.
        :param labels: (tf_graph_tensor) The corresponding labels to the data.
        :return: distances_per_layer: (tf_graph_tensors) The representation of each layer. The last layers
                                        representation is the csnn representation used for tasks like classification
        :return: patches_out: (tf_graph_tensors) Returns the image_patches of each layer for visualization.
        :return: labels : (tf_graph_tensors) If the data is labeled, the labels for the distances a
                            nd patches will be returned.
        """
        distances_per_layer = []
        patches_out = []

        # Split the batch for each tower.
        input_data_split = tf.split(input_data,  1)

        # Get the current variable scope to reuse all variables in the second iteration of the loop below.
        with tf.variable_scope(tf.get_variable_scope()) as outer_scope:
            # After the first iteration, reuse the variables.
            outer_scope.reuse_variables()
            for gpu in range(1):
                with tf.name_scope('ValidationGPU%d' % gpu), tf.device('/gpu:%d' % gpu):
                    distances_per_layer_batch, patches_out_batch = model.getValOp(input_data_split[gpu])
                distances_per_layer.append(distances_per_layer_batch)
                patches_out.append(patches_out_batch)
        return distances_per_layer, patches_out, labels

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
        # Split the batch for each tower.
        input_data_split = tf.split(input_data, self.__num_gpus)
        train_ops = []
        # Get the current variable scope to reuse all variables in the second iteration of the loop below.
        with tf.variable_scope(tf.get_variable_scope()) as outer_scope:
            for gpu in range(self.__num_gpus):
                with tf.name_scope('TrainingGPU%d' % gpu), tf.device('/gpu:%d' % gpu):
                    # Compute new_batch_weights, but don't apply them yet
                    with tf.name_scope("computeLearningWeights"):
                        train_ops.append(model.getTrainOp(input_data_split[gpu]))

                # After the first iteration, reuse the variables.
                outer_scope.reuse_variables()

        # Apply the learned weights on the controlling device
        with tf.name_scope("applyLearningWeights"), tf.device(self.__controller):
            global_step = tf.train.get_or_create_global_step()
            tf.summary.scalar('global_step', global_step)
            apply_learning_weights, increment_global_step_op = model.applySomLearning(global_step, self.__num_gpus)
        return train_ops, apply_learning_weights, increment_global_step_op


