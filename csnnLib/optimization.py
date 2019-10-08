import tensorflow as tf

"""
Defines optimization functions of the CSNN, to update the different weights of the layers.
"""


def optimize_csnn(global_step, num_gpus, layer_train_intervals, verbose=True):
    """
    Optimizes the entire CSNN model (som and mask weights) in the graph by getting the weight collections.

    :param global_step: (Float) The current global step.
    :param num_gpus: (Float) The number of GPUs used.
    :param layer_train_intervals: (Array) The training intervals of the layers. (E.g. [[0,10000],[10000,20000]])
    :param verbose: (Boolean) If true plot statistics of the optimization. True by default.
    :return: [apply_weights_ops, apply_local_weights_ops] or apply_weights_ops: (Tensors) The operations to apply
             the weight changes of all layers from all GPUS for the som weights and the mask weights (when used).
    :return: inc_global_step_op: (Tensor) The operation to increment the global step.
    """
    with tf.variable_scope('OptimizeCsnn'):
        # Get the som optimization operation for each layer.
        apply_weights_ops = __optimize_som_weights(num_gpus, layer_train_intervals, verbose)

        # Define the global step incrementation operation.
        inc_global_step_op = tf.assign(global_step, global_step + 1)

        # If there are local weights (masks) get the mask optimization operations for each layer and return them too.
        if tf.get_collection("deltaLocalWeights"):
            apply_local_weights_ops = __optimize_local_weights(num_gpus, layer_train_intervals, verbose)
            return [apply_weights_ops, apply_local_weights_ops], inc_global_step_op
        # Else return only the som weight optimization and the increment operation of the global step.
        else:
            return apply_weights_ops, inc_global_step_op


def __optimize_som_weights(num_gpus, layer_train_intervals, verbose):
    """
    Creates the optimization operation from the som weights of each layer from each GPU.
    :param num_gpus: (Float) The number of GPUs used.
    :param layer_train_intervals: (Array) The training intervals of the layers. (E.g. [[0,10000],[10000,20000]])
    :param verbose: (Boolean) If true plot statistics of the optimization.
    :return: apply_som_weights_ops: The operations to apply the weight changes of all layers from all GPUS for the
            som weights
    """
    with tf.variable_scope('OptimizeSomWeights'):
        # Get statistics if verbose.
        if verbose:
            distance_measures = tf.get_collection("sum_distance_measure")
            neuron_utilization = tf.get_collection("neuronUtilization")

        # Get the delta weights and the weight variables of each layer of the CSNN.
        delta_som_weights = tf.get_collection("deltaSomWeights")
        old_som_weights = tf.get_collection("oldSomWeights")

        # Calculate the number of layers of the CSNN.
        num_layers = len(old_som_weights) // num_gpus

        apply_som_weights_ops = []
        # For each layer
        for i in range(0, num_layers):
            gpu_delta_weights = []
            gpu_distance_measures = []
            gpu_neuron_utilization = []

            # Collect all weight changes from the GPUs.
            for j in range(i, len(old_som_weights), num_layers):
                gpu_delta_weights.append(delta_som_weights[j])
                # If verbose collect all statistics.
                if verbose:
                    if distance_measures:
                        gpu_distance_measures.append(distance_measures[j])
                    if neuron_utilization:
                        gpu_neuron_utilization.append(neuron_utilization[j])

            # If verbose updated the summary with statistics.
            if verbose:
                tf.summary.scalar("MeanWeightChangeLayer" + str(i), tf.reduce_mean(gpu_delta_weights))
                tf.summary.histogram("SomWeightsLayer" + str(i), old_som_weights[i])

                if distance_measures:
                    tf.summary.scalar("DistanceMeasure" + str(i), tf.reduce_mean(gpu_distance_measures))
                if neuron_utilization:
                    tf.summary.scalar("NeuronUtilization" + str(i), tf.reduce_mean(gpu_neuron_utilization))
                    print(gpu_neuron_utilization)

            def train():
                # The operation to apply the som weight changes for each layer.
                return applyStandardSomLearning(old_som_weights[i], gpu_delta_weights)

            def infer():
                return 0.0

            # Only train the weights of the layer, if the global step lies in the layers training interval.
            global_step = tf.train.get_or_create_global_step()
            operation = tf.cond(
                tf.logical_and(
                    tf.greater_equal(tf.cast(global_step, tf.int32), tf.cast(layer_train_intervals[i][0], tf.int32)),
                    tf.less(tf.cast(global_step, tf.int32), tf.cast(layer_train_intervals[i][1], tf.int32))),
                true_fn=train,
                false_fn=infer)
            apply_som_weights_ops.append(operation)

        # Return the operation to apply the weight changes for each layer.
        return apply_som_weights_ops


def __optimize_local_weights(num_gpus, layer_train_intervals, verbose):
    """
    Creates the optimization operation from the local (mask) weights of each layer from each GPU.
    :param num_gpus: (Float) The number of GPUs used.
    :param layer_train_intervals: (Array) The training intervals of the layers. (E.g. [[0,10000],[10000,20000]])
    :param verbose: (Boolean) If true plot statistics of the optimization.
    :return: apply_weights_ops: The operations to apply the weight changes of all layers from all GPUS for the
            local (mask= weights
    """
    with tf.variable_scope('OptimizeLocalWeights'):
        # Get the delta local weights and the local weight variables of each layer of the CSNN.
        delta_local_weights = tf.get_collection("deltaLocalWeights")
        old_local_weights = tf.get_collection("oldLocalWeights")

        # Calculate the number of layers of the CSNN.
        num_mask_layers = len(old_local_weights) // num_gpus

        apply_local_weights_ops = []
        # For each layer
        for i in range(0, num_mask_layers):
            gpu_delta_local_weights = []
            # Collect all local weight changes from the GPUs.
            for j in range(i, len(old_local_weights), num_mask_layers):
                gpu_delta_local_weights.append(delta_local_weights[j])

            # If verbose updated the summary with statistics..
            if verbose:
                tf.summary.scalar("MeanLocalWeightChangeLayer" + str(i), tf.reduce_mean(gpu_delta_local_weights))
                tf.summary.histogram("LocalWeightsLayer" + str(i), old_local_weights[i])

            def train():
                # The operation to apply the local weight changes for each layer.
                return applyLocalLearning(old_local_weights[i], gpu_delta_local_weights)

            def infer():
                return 0.0

            # Only train the weights of the layer, if the global step lies in the layers training interval.
            global_step = tf.train.get_or_create_global_step()
            operation = tf.cond(
                tf.logical_and(tf.greater_equal(tf.cast(global_step, tf.int32), tf.cast(layer_train_intervals[i][0], tf.int32)),
                               tf.less(tf.cast(global_step, tf.int32), tf.cast(layer_train_intervals[i][1], tf.int32))),
                true_fn=train,
                false_fn=infer)
            apply_local_weights_ops.append(operation)

        # Return the operation to apply the local weight changes for each layer.
        return apply_local_weights_ops


def applyStandardSomLearning(som_net_weights, delta_new_batch_weights):
    """
    Applies the weight change of a layer by taking the mean form all GPU weight changes.
    :param som_net_weights:  (Tensor) The weights to change
                             Shape: [num_maps, num_som_neurons, weight_vector_size]
    :param delta_new_batch_weights: (Tensor) The weight changes from each GPU.
                                     Shape: [num_gpus, num_maps, num_som_neurons, weight_vector_size]
    :return: apply_weights_op: (Tensor) The operation to apply the weight change.
    """
    with tf.variable_scope("applyStandardSomLearning"):
        # Compute the new weights of each som neuron by taking the mean across the GPUs.
        # Input Tensor Shape: [num_gpus, num_maps, num_som_neurons, weight_vector_size]
        # Output Tensor Shape: [num_maps, num_som_neurons, weight_vector_size]
        delta_new_weights = tf.reduce_mean(delta_new_batch_weights, axis=0)

        # Compute the new weights of each som neuron by adding the delta weight.
        new_weights = tf.add(som_net_weights, delta_new_weights)

        # Normalize the weights.
        new_weights = tf.div_no_nan(new_weights, tf.norm(new_weights, axis=2, keepdims=True))

        # Apply the new weights.
        apply_weights_op = tf.assign(som_net_weights, new_weights)

        return apply_weights_op


def applyLocalLearning(local_weights, delta_new_local_weights):
    """
    Applies the local weight change of a layer by taking the mean form all GPU weight changes.
    :param local_weights:    (Tensor) The weights to change
                             Shape: [num_maps, num_som_neurons, local_weight_size]
    :param delta_new_local_weights: (Tensor) The weight changes from each GPU.
                                     Shape: [num_gpus, num_maps, num_som_neurons, local_weight_size]
    :return: apply_local_weights_op: (Tensor) The operation to apply the weight change.
    """
    with tf.variable_scope("applyLocalLearning"):
        # Compute the new local weights for each som neuron by taking the mean across the GPUs.
        # Input Tensor Shape: [num_gpus, num_maps, num_som_neurons, local_weight_size]
        # Output Tensor Shape: [num_maps, num_som_neurons, num_som_neurons_prev_layer]
        delta_new_weights = tf.reduce_mean(delta_new_local_weights, axis=0)

        # Compute the new local weights of each som neuron by adding the delta weight.
        new_weights = tf.add(local_weights, delta_new_weights)

        # Apply the new local weights.
        apply_local_weights_op = tf.assign(local_weights, new_weights)

        return apply_local_weights_op
