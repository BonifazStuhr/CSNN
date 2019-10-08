import tensorflow as tf

from Experiment_Component.ITrainer import ITrainer

class SingleGpuBackprobTrainer(ITrainer):
    """
    The SingleGpuBackprobTrainer trains a model via backpropagation algorithms on a single gpu.
    """
    def createValidation(self, model, input_data, labels):
        """
        Creates the validation of the model.
        :param model: (tf_graph_tensor) The model to validate.
        :param input_data: (tf_graph_tensor) The data to validate the model.
        :param labels: (tf_graph_tensor) The corresponding labels to the data.
        :return: loss_op: (tf_graph_tensors) The operations to calculate the loss of the current validation step.
        :return: acc_op: (tf_graph_tensors) The operations to calculate the accuracy of the current validation step.
        """
        with tf.variable_scope(tf.get_variable_scope()) as outer_scope:
            with tf.name_scope('Validation'):
                # Reuse the current variables of the model.
                outer_scope.reuse_variables()
                loss_op, logits_op = model.getValLossOp(input_data, labels)
                acc_op = model.getEvaluationOp(logits_op, labels)
                return loss_op, acc_op

    def createTraining(self, model, input_data, labels):
        """
        Creates the training of the model.
        :param model: (tf_graph_tensor) The model to train.
        :param input_data: (tf_graph_tensor) The data to train the model.
        :param labels: (tf_graph_tensor) The corresponding labels to the data.
        :return: update_op: (tf_graph_tensors) The operations to execute the training in a tf.session.
        :return: loss_op: (tf_graph_tensors) The operations to calculate the loss of the current training step.
        :return: acc_op: (tf_graph_tensors) The operations to calculate the accuracy of the current training step.
        """
        with tf.variable_scope(tf.get_variable_scope()) as outer_scope:
            with tf.name_scope('Training'):
                loss_op, logits_op = model.getTrainLossOp(input_data, labels)
                acc_op = model.getEvaluationOp(logits_op, labels)
                global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False,
                                      dtype=tf.float32)
                update_op = model.getOptimizerOp(global_step).minimize(loss_op, global_step=global_step)
                return update_op, loss_op, acc_op


