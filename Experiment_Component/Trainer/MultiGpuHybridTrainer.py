import tensorflow as tf

from Experiment_Component.ITrainer import ITrainer

class MultiGpuHybridTrainer(ITrainer):
    """
    The MultiGpuHybridTrainer trains a CNN-CSNN-Hybrid model on multiple GPUs.

     :Attributes:
        __num_gpus:    (Integer) The number of gpus to use.
        __controller:  (String) The name of the controller. The controller is the CPU/GPU where the gradients are
                                 accumulated and applied to the model.
    """
    def __init__(self, num_gpus, controller="/cpu:0"):
        """
        Constructor, initialize member variables.
        :param num_gpus: (Integer) The number of gpus to use.
        :param controller: (String) The name of the controller. The controller is the CPU/GPU where the gradients are
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
        :return: avg_loss_op: (tf_graph_tensors) The operations to calculate the loss of the current validation step.
        :return: avg_acc_op: (tf_graph_tensors) The operations to calculate the accuracy of the current validation step.
        """
        # This lists keeps track of the losses and accs per tower.
        losses = []
        logits = []
        csnn_eval_ops = []

        # Split the batch for each tower.
        input_data_split = tf.split(input_data, self.__num_gpus)
        labels_split = tf.split(labels, self.__num_gpus)

        # Get the current variable scope to reuse all variables we need once we get to the second iteration of the loop
        # below.
        with tf.variable_scope(tf.get_variable_scope()) as outer_scope:
            outer_scope.reuse_variables()
            for gpu in range(self.__num_gpus):
                with tf.name_scope('ValidationGPU%d' % gpu), tf.device('/gpu:%d' % gpu):
                    # Compute loss and logits.
                    gpu_loss, gpu_logits, gpu_csnn_distances = model.getValLossOp(input_data_split[gpu], labels_split[gpu])
                losses.append(gpu_loss)
                logits.append(gpu_logits)
                csnn_eval_ops.append(gpu_csnn_distances)

        # Calculate loss and acc on the controlling device
        with tf.name_scope("valLossAcc"), tf.device(self.__controller):
            avg_loss_op = tf.reduce_mean(losses)
            avg_acc_op = model.getEvaluationOp(logits, labels)
        return avg_loss_op, avg_acc_op, csnn_eval_ops

    def createTraining(self, model, input_data, labels):
        """
        Creates the training of the model.
        :param model: (tf_graph_tensor) The model to train.
        :param input_data: (tf_graph_tensor) The data to train the model.
        :param labels: (tf_graph_tensor) The corresponding labels to the data.
        :return: apply_update_ops: (tf_graph_tensors) The operations to execute the training in a tf.session.
        :return: avg_loss_op: (tf_graph_tensors) The operations to calculate the loss of the current training step.
        :return: avg_acc_op: (tf_graph_tensors) The operations to calculate the accuracy of the current training step.
        """
        # This list keeps track of the gradients, updates, losses and accs per tower.
        tower_grads = []
        csnn_train_ops = []
        losses = []
        logits = []

        # Split the batch for each tower.
        input_data_split = tf.split(input_data, self.__num_gpus)
        labels_split = tf.split(labels, self.__num_gpus)

        # Get the current variable scope to reuse all variables in the second iteration of the loop below.
        with tf.variable_scope(tf.get_variable_scope()) as outer_scope:
            global_step = tf.train.get_or_create_global_step()
            for gpu in range(self.__num_gpus):
                with tf.name_scope('TrainingGPU%d' % gpu), tf.device('/gpu:%d' % gpu):
                    # Compute loss and gradients and csnn_distances, but don't apply changes yet
                    gpu_loss, gpu_logits, gpu_csnn_distances = model.getTrainLossOp(input_data_split[gpu], labels_split[gpu])
                    with tf.name_scope("computeGradients"):
                        # compute_gradients returns a list of (gradient, variable) pairs
                        gpu_grads = model.getCnnOptimizerOp(global_step).compute_gradients(gpu_loss)
                        tower_grads.append(gpu_grads)

                    csnn_train_ops.append(gpu_csnn_distances)
                    losses.append(gpu_loss)
                    logits.append(gpu_logits)

                # After the first iteration, reuse the variables.
                outer_scope.reuse_variables()

        # Apply the gradients and csnn updates on the controlling device
        with tf.name_scope("applyGradientsAndCsnnUpdates"), tf.device(self.__controller):
            gradients = self.__averageGradients(tower_grads)
            apply_gradient_op = model.getCnnOptimizerOp(global_step).apply_gradients(gradients, global_step)
            gpu_csnn_updates = model.getCsnnOptimizerOp(global_step)
            # Calculate loss and acc
            avg_loss_op = tf.reduce_mean(losses)
            avg_acc_op = model.getEvaluationOp(logits, labels)

        return csnn_train_ops, apply_gradient_op, gpu_csnn_updates, avg_loss_op, avg_acc_op

    def __averageGradients(self, tower_grads):
        """
        Calculate the average gradient for each shared variable across all towers.
        Note that this function provides a synchronization point across all towers.
        :param tower_grads: (List of lists of (gradient, variable) tuples) The outer list ranges over the devices.
                             The inner list ranges over the different variables.
        :return: average_grads: (List of pairs of (gradient, variable)) where the gradient has been averaged across all
                                    towers.
        """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Each grad_and_vars looks like the following: ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN)).
            grads = [g for g, _ in grad_and_vars]
            if grads[0] is None:
                continue
            grad = tf.reduce_mean(grads, 0)

            # The variables are redundant because they are shared across towers.
            # So just return the first tower's pointer to the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads
