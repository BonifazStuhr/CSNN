import tensorflow as tf

"""
Defines learning rules for the som weights and local weights (masks) of the CSNN.
"""

som_learning_rule_names = ["standardSomLearning", "convSomLearning", "statics"]
local_learning_rule_names = [0,
                             "hebb",
                             "oja",
                             "proposedLocalLearningHebb",
                             "proposedLocalLearningHebb05",
                             "proposedLocalLearningOja",
                             "generalizedHebbianElemMul",
                             "generalizedHebbianElemMul05",
                             "generalizedHebbianElemMulNoWeight",
                             "generalizedHebbianElemMulNoWeight05",
                             "noise",
                             "static"
                             ]

def standardSomLearning(learning_rate, neighborhood_coeff_batch, input_batch, som_net_weights):
    """
    Calculates the batch weight change of a layer of the som weights from SOMs with Euclidean distance.
    :param learning_rate: (Float) The learning rate for the weight change.
    :param neighborhood_coeff_batch: (Tensor) The neighborhood_coeff for each neuron.
                                Shape: [batch_size, num_maps, num_som_neurons]
    :param input_batch: (Float) The input batch of the layer.
                                Shape: [batch_size, 1 or num_maps, 1 or num_som_neurons, som_neuron_weight_vector_size]
    :param som_net_weights:  The som_net_weights to learn. Not used.
                             Shape: [num_maps, num_som_neurons, som_neuron_weight_vector_size]
    :return: delta_new_weights_op: (Tensor) The operation to learn the weights.
                                    Shape: [num_maps, num_som_neurons, som_neuron_weight_vector_size]
    """
    with tf.variable_scope("standardSomLearning"):

        # Input Tensor Shape: [batch_size, num_maps, num_som_neurons]
        # Output Tensor Shape: [batch_size, num_maps, num_som_neurons, 1]
        neighborhood_coeff_batch = tf.expand_dims(neighborhood_coeff_batch, axis=3)

        # Compute the new weights of each som neuron for each batch
        # Input Tensor Shape input_batch:
        # [batch_size, 1 or num_maps, 1 or num_som_neurons, som_neuron_weight_vector_size]
        # Input Tensor Shape som_net_weights: [num_maps, num_som_neurons, som_neuron_weight_vector_size]
        # Input Tensor Shape neighborhood_coeff_batch: [batch_size, num_maps, num_som_neurons, 1]
        # After expand dims Tensor Shape: [1, num_maps, num_som_neurons, som_neuron_weight_vector_size]
        # After subtract Tensor Shape: [1, num_maps, num_som_neurons, som_neuron_weight_vector_size]
        # Output Tensor Shape: [batch_size, num_maps, num_som_neurons, som_neuron_weight_vector_size]
        # - the weights for each som neuron and each batch
        delta_new_batch_weights = tf.multiply(neighborhood_coeff_batch,
                                              tf.subtract(input_batch, tf.expand_dims(som_net_weights, axis=0)))

        # Compute the new weights of each som neuron by taking the mean across the batches.
        # Input Tensor Shape: [batch_size, num_maps, num_som_neurons, som_neuron_weight_vector_size]
        # Output Tensor Shape: [num_maps, num_som_neurons, som_neuron_weight_vector_size]
        delta_new_weights = tf.reduce_mean(delta_new_batch_weights, axis=0)

        # Compute the delta_new_weights multiplied with the learning rate.
        # Input Tensor Shape delta_new_weights: [num_maps, num_som_neurons, som_neuron_weight_vector_size]
        # Input Tensor Shape learn_rate: [1]
        # Output Tensor Shape: [num_maps, num_som_neurons, som_neuron_weight_vector_size]
        delta_new_weights_op = tf.multiply(delta_new_weights, learning_rate)

        # Output Tensor Shape: [num_maps, num_som_neurons, som_neuron_weight_vector_size]
        return delta_new_weights_op


def convSomLearning(learning_rate, neighborhood_coeff_batch, input_batch, som_net_weights=None):
    """
    Calculates the batch weight change of a layer of the som weights from SOms with convolution as described in the
    paper.
    :param learning_rate: (Float) The learning rate for the weight change
    :param neighborhood_coeff_batch: (Tensor) The neighborhood_coeff for each neuron.
                                Shape: [batch_size, num_maps, num_som_neurons]
    :param input_batch: (Float) The input batch of the layer.
                                Shape: [batch_size, 1 or num_maps, 1 or num_som_neurons, som_neuron_weight_vector_size]
    :param som_net_weights:  The som_net_weights to learn. Not used.
                             Shape: [num_maps, num_som_neurons, som_neuron_weight_vector_size]
    :return: delta_new_weights_op: (Tensor) The operation to learn the weights.
                                    Shape: [num_maps, num_som_neurons, som_neuron_weight_vector_size]
    """
    with tf.variable_scope("convSomLearning"):

        # Input Tensor Shape: [batch_size, num_maps, num_som_neurons]
        # Output Tensor Shape: [batch_size, num_maps, num_som_neurons, 1]
        neighborhood_coeff_batch = tf.expand_dims(neighborhood_coeff_batch, axis=3)

        # Compute the new weights of each som neuron for each batch
        # Input Tensor Shape input_batch:
        # [batch_size, 1 or num_maps, 1 or num_som_neurons, som_neuron_weight_vector_size]
        # Input Tensor Shape neighborhood_coeff_batch: [batch_size, num_maps, num_som_neurons, 1]
        # Output Tensor Shape : [batch_size, num_maps, num_som_neurons, som_neuron_weight_vector_size]
        delta_new_batch_weights_batch = tf.multiply(neighborhood_coeff_batch, input_batch)

        # Compute the new weights of each som neuron by taking the mean across the batches.
        # Input Tensor Shape: [batch_size, num_maps, num_som_neurons, som_neuron_weight_vector_size]
        # Output Tensor Shape: [num_maps, num_som_neurons, som_neuron_weight_vector_size]
        # - the new delta_weights for each som neuron
        delta_new_weights = tf.reduce_mean(delta_new_batch_weights_batch, axis=0)

        # Compute the delta_new_weights multiplied with the learning rate.
        # Input Tensor Shape delta_new_weights: [num_maps, num_som_neurons, som_neuron_weight_vector_size]
        # Input Tensor Shape learn_rate: [1]
        # Output Tensor Shape: [num_maps, num_som_neurons, som_neuron_weight_vector_size]
        delta_new_weights_op = tf.multiply(delta_new_weights, learning_rate)

        # Output Tensor Shape: [num_maps, num_som_neurons, som_neuron_weight_vector_size]
        return delta_new_weights_op


def statics(learning_rate, neighborhood_coeff_batch, input_batch, som_net_weights=None):
    """
    Stets the delta weights to zero.
    :param learning_rate: (Float) The learning rate for the weight change
    :param neighborhood_coeff_batch: (Tensor) The neighborhood_coeff for each neuron.
                                Shape: [batch_size, num_maps, num_som_neurons]
    :param input_batch: (Float) The input batch of the layer.
                                Shape: [batch_size, 1 or num_maps, 1 or num_som_neurons, som_neuron_weight_vector_size]
    :param som_net_weights:  The som_net_weights to learn. Not used.
                             Shape: [num_maps, num_som_neurons, som_neuron_weight_vector_size]
    :return: delta_new_weights_op: (Tensor) The operation to learn the weights.
                                    Shape: [num_maps, num_som_neurons, som_neuron_weight_vector_size]
    """
    with tf.variable_scope("statics"):
        # Output Tensor Shape: [num_maps, num_som_neurons, som_neuron_weight_vector_size]
        m, n, d = som_net_weights.get_shape()
        delta_new_weights_op = tf.zeros([m.value, n.value, d.value], dtype=tf.float32)

        # Output Tensor Shape: [num_maps, num_som_neurons, som_neuron_weight_vector_size]
        return delta_new_weights_op


def __prepareForLocalWeightsLearning(neighborhood_coeff_batch, local_neighborhood_coeff_type, local_weights_type):
    """
    Prepares the given input for the calculation of delta weights for the given local weight type.
    For local weights between input and neurons we can set the reduce mean axis for the batch to the batch dimension
    to gain the delta weights.
    For local weights between neurons we can set the reduce mean axis for the batch to the batch and kernel dimension.
    to gain the delta weights.
    For local weights between input and neurons we can set the reduce sum axis for to calculate the manipulated input
    to the input dimension.
    For local weights between input and neurons we can set the reduce sum axis for to calculate the manipualated input
    to the input dimension.
    We need to expand the neighborhood_coeff if used. All means every mask is updated, somNeighborhood means we take the
    som neighborhood for local updates, if none of these types is given, we only update the bmu mask.
    :param neighborhood_coeff_batch: (Tensor) The neighborhood coefficients for Oja to calculate the local weights.
                                     of the weights, restricting the update for particular weights if =! 1.
                                     Shape: [batch_size, num_maps, num_som_neurons].
    :param local_neighborhood_coeff_type: (String) The type of neighborhood to use for the update.
    :param local_weights_type: (String) The type of local weights to update.
    :return: neighborhood_coeff_batch: (Tensor) The neighborhood coefficients for Oja to calculate the local weights.
                                     of the weights, restricting the update for particular weights if =! 1.
                                     Shape: [batch_size, (1), num_maps, num_som_neurons, 1] or [1].
    :return: reduce_sum_axis: (Array or Integer) The input axis to reduce in the learning.
    :return: reduce_mean_axis: (Array or Integer) The batch axis to reduce in the learning.
    """
    reduce_sum_axis = ""
    reduce_mean_axis = ""
    if local_neighborhood_coeff_type == "all":
        return 1.0
    elif local_neighborhood_coeff_type == "somNeighborhood":
        neighborhood_coeff_batch = tf.expand_dims(neighborhood_coeff_batch, axis=3)
    else:
        neighborhood_coeff_batch = tf.to_float(tf.greater_equal(neighborhood_coeff_batch, 1.0))
        neighborhood_coeff_batch = tf.expand_dims(neighborhood_coeff_batch, axis=3)

    if local_weights_type == "betweenNeurons":
        reduce_sum_axis = 3
        reduce_mean_axis = [0, 1]
        neighborhood_coeff_batch = tf.expand_dims(neighborhood_coeff_batch, axis=1)
    elif local_weights_type == "betweenInputAndNeurons":
        reduce_sum_axis = 2
        reduce_mean_axis = [0]

    return neighborhood_coeff_batch, reduce_sum_axis, reduce_mean_axis


def oja(local_weights, input_local_update_batch, neighborhood_coeff_batch, output_local_update_batch,
                       local_learning_rate, reduce_sum_axis, reduce_mean_axis):
    """
    Calculates the delta weights following Oja's Rule (d_w = a*y*(x-y*w)) element-wise for the given input and
    output batch and weigths. The elemet-wise delta weights are located in each spatial entry of the output and are
    the mean over every sample in the batch.
    :param local_weights: (Tensor) The local weight to calculate the delta weights for.
                        Shape: [num_maps, num_som_neurons, num_som_neurons_prev_layer or som_neuron_weight_vector_size]
    :param input_local_update_batch: (Tensor) The input_batch for the Rule to calculate the local weights.
                                     Shape: [batch_size, (kernel_width*kernel_height), 1, 1,
                                     num_som_neurons_prev_layer or som_neuron_weight_vector_size]
    :param neighborhood_coeff_batch: (Tensor) The neighborhood coefficients for Oja to calculate the local weights.
                                      of the weights, restricting the update for particular weights if =! 1.
                                      Shape: [batch_size, (1), num_maps, num_som_neurons, 1] or [1]
    :param output_local_update_batch: (Tensor) The local output_batch for theRule to calculate the local weights.
                                      Shape: [batch_size, (kernel_width*kernel_height), num_maps, num_som_neurons,
                                      num_som_neurons_prev_layer or som_neuron_weight_vector_size]
    :param local_learning_rate: (Float) The learning rate for the Rule.
    :param reduce_sum_axis: (Integer) The axis to reduce in the input to support bot local weight types. Not used here!
    :param reduce_mean_axis: (Array) The axis to reduce the batch for the delta weights to support bot
                              local weight types.
    :return: delta_local_weights_op: (Tensor) The operation to calculate the delta weights.
    """
    with tf.variable_scope("Oja"):
        # Input Tensor Shape local_weights: [num_maps, num_som_neurons, num_som_neurons_prev_layer or
        # som_neuron_weight_vector_size]
        # Input Tensor Shape input_local_update_batch: [batch_size, (kernel_width*kernel_height), 1, 1,
        # num_som_neurons_prev_layer or num_som_neurons_prev_layer]
        # Input Tensor Shape neighborhood_coeff_batch: [batch_size, (1), num_maps, num_som_neurons, 1]
        # Input Tensor Shape output_local_update_batch: [batch_size, kernel_width*kernel_height, num_maps,
        # num_som_neurons, num_som_neurons_prev_layer or num_som_neurons_prev_layer]
        # Output Tensor Shape: [batch_size, (kernel_width*kernel_height), num_maps, num_som_neurons,
        # num_som_neurons_prev_layer or num_som_neurons_prev_layer]
        delta_weights_batch = tf.multiply(neighborhood_coeff_batch, tf.multiply(output_local_update_batch,
                                            tf.subtract(input_local_update_batch, tf.multiply(output_local_update_batch,
                                                                                             local_weights))))

        # Input Tensor Shape: [batch_size, (kernel_width*kernel_height), num_maps, num_som_neurons,
        # num_som_neurons_prev_layer or som_neuron_weight_vector_size]
        # Output Tensor Shape: [num_maps, num_som_neurons, num_som_neurons_prev_layer or som_neuron_weight_vector_size]
        delta_local_weights = tf.reduce_mean(delta_weights_batch, axis=reduce_mean_axis)

        # Input Tensor Shape: [num_maps, num_som_neurons, num_som_neurons_prev_layer or som_neuron_weight_vector_size]
        # Input Tensor Shape: [1]
        # Output Tensor Shape: [num_maps, num_som_neurons, num_som_neurons_prev_layer or som_neuron_weight_vector_size]
        delta_local_weights_op = tf.multiply(delta_local_weights, local_learning_rate)

        # Output Tensor Shape: [num_maps, num_som_neurons, num_som_neurons_prev_layer or som_neuron_weight_vector_size]
        return delta_local_weights_op


def hebb(local_weights, input_local_update_batch, neighborhood_coeff_batch, output_local_update_batch,
        local_learning_rate, reduce_sum_axis, reduce_mean_axis):
    """
    Calculates the delta weights following Hebbs's Rule (d_w = a*y*x) element-wise for the given input and
    output batch and weigths. The elemet-wise delta weights are located in each spatial entry of the output and are
    the mean over every sample in the batch.
    :param local_weights: (Tensor) The local weight to calculate the delta weights for. Not used here!
                        Shape: [num_maps, num_som_neurons, num_som_neurons_prev_layer or som_neuron_weight_vector_size]
    :param input_local_update_batch: (Tensor) The input_batch for the Rule to calculate the local weights.
                                     Shape: [batch_size, (kernel_width*kernel_height), 1, 1,
                                     num_som_neurons_prev_layer or som_neuron_weight_vector_size]
    :param neighborhood_coeff_batch: (Tensor) The neighborhood coefficients for Oja to calculate the local weights.
                                     of the weights, restricting the update for particular weights if =! 1.
                                     Shape: [batch_size, (1), num_maps, num_som_neurons, 1] or [1]
    :param output_local_update_batch: (Tensor) The local output_batch for theRule to calculate the local weights.
                                      Shape: [batch_size, (kernel_width*kernel_height), num_maps, num_som_neurons,
                                      num_som_neurons_prev_layer or som_neuron_weight_vector_size]
    :param local_learning_rate: (Float) The learning rate for the Rule.
    :param reduce_sum_axis: (Integer) The axis to reduce in the input to support bot local weight types. Not used here!
    :param reduce_mean_axis: (Array) The axis to reduce the batch for the delta weights to support bot
                             local weight types.
    :return: delta_local_weights_op: (Tensor) The operation to calculate the delta weights.
    """
    with tf.variable_scope("hebb"):
        # Input Tensor Shape input_local_update_batch: [batch_size, (kernel_width*kernel_height), 1, 1,
        # num_som_neurons_prev_layer or num_som_neurons_prev_layer]
        # Input Tensor Shape neighborhood_coeff_batch: [batch_size, (1), num_maps, num_som_neurons, 1]
        # Input Tensor Shape output_local_update_batch: [batch_size, kernel_width*kernel_height, num_maps,
        # num_som_neurons, num_som_neurons_prev_layer or num_som_neurons_prev_layer]
        # Output Tensor Shape: [batch_size, (kernel_width*kernel_height), num_maps, num_som_neurons,
        # num_som_neurons_prev_layer or num_som_neurons_prev_layer]
        delta_weights_batch = tf.multiply(neighborhood_coeff_batch, tf.multiply(output_local_update_batch,
                                                                              input_local_update_batch))

        # Input Tensor Shape: [batch_size, (kernel_width*kernel_height), num_maps, num_som_neurons,
        # num_som_neurons_prev_layer or som_neuron_weight_vector_size]
        # Output Tensor Shape: [num_maps, num_som_neurons, num_som_neurons_prev_layer or som_neuron_weight_vector_size]
        delta_local_weights = tf.reduce_mean(delta_weights_batch, axis=reduce_mean_axis)

        # Input Tensor Shape: [num_maps, num_som_neurons, num_som_neurons_prev_layer or som_neuron_weight_vector_size]
        # Input Tensor Shape: [1]
        # Output Tensor Shape: [num_maps, num_som_neurons, num_som_neurons_prev_layer or som_neuron_weight_vector_size]
        delta_local_weights_op = tf.multiply(delta_local_weights, local_learning_rate)

        # Output Tensor Shape: [num_maps, num_som_neurons, num_som_neurons_prev_layer or som_neuron_weight_vector_size]
        return delta_local_weights_op

def proposedLocalLearningHebb(local_weights, input_local_update_batch, neighborhood_coeff_batch,
                              output_local_update_batch, local_learning_rate, reduce_sum_axis, reduce_mean_axis,
                              input_modification_coeff=1.0):
    """
    Calculates the delta weights following Our Rule (12) in the paper (d_w = a*y*x), where x=x-r*sum_k(y*w))
    element-wise for the given input and output batch and weigths. The elemet-wise delta weights are located in each
    spatial entry of the output and are the mean over every sample in the batch.
    :param local_weights: (Tensor) The local weight to calculate the delta weights for.
                        Shape: [num_maps, num_som_neurons, num_som_neurons_prev_layer or som_neuron_weight_vector_size]
    :param input_local_update_batch: (Tensor) The input_batch for the Rule to calculate the local weights.
                                     Shape: [batch_size, (kernel_width*kernel_height), 1, 1,
                                     num_som_neurons_prev_layer or som_neuron_weight_vector_size]
    :param neighborhood_coeff_batch: (Tensor) The neighborhood coefficients for Oja to calculate the local weights.
                                     of the weights, restricting the update for particular weights if =! 1.
                                     Shape: [batch_size, (1), num_maps, num_som_neurons, 1] or [1]
    :param output_local_update_batch: (Tensor) The local output_batch for theRule to calculate the local weights.
                                      Shape: [batch_size, (kernel_width*kernel_height), num_maps, num_som_neurons,
                                      num_som_neurons_prev_layer or som_neuron_weight_vector_size]
    :param local_learning_rate: (Float) The learning rate for the Rule.
    :param reduce_sum_axis: (Integer) The axis to reduce in the input to support bot local weight types.
    :param reduce_mean_axis: (Array) The axis to reduce the batch for the delta weights to support bot
                             local weight types.
    :param input_modification_coeff: (Integer) The coeff for the input modification. 1.0 by default.
    :return: delta_local_weights_op: (Tensor) The operation to calculate the delta weights.
    """
    with tf.variable_scope("proposedLocalLearningHebb"):
        # Input Tensor Shape local_weights: [num_maps, num_som_neurons, num_som_neurons_prev_layer or
        # som_neuron_weight_vector_size]
        # Input Tensor Shape input_local_update_batch: [batch_size, (kernel_width*kernel_height), 1, 1,
        # num_som_neurons_prev_layer]
        # Input Tensor Shape neighborhood_coeff_batch: [batch_size, (1), num_maps, num_som_neurons]
        # Input Tensor Shape output_local_update_batch: [batch_size, kernel_width*kernel_height, num_maps,
        # num_som_neurons, num_som_neurons_prev_layer]
        # Input Tensor Shape input_modification_coeff: [1]
        # Output Tensor Shape output_local_update_batch: [batch_size, kernel_width*kernel_height, num_maps,
        # 1, num_som_neurons_prev_layer]
        modified_input_batch = tf.subtract(input_local_update_batch, tf.multiply(input_modification_coeff,
                                           tf.reduce_sum(tf.multiply(output_local_update_batch, local_weights),
                                           axis=reduce_sum_axis, keepdims=True)))

        # Input Tensor Shape local_weights: [num_maps, num_som_neurons, num_som_neurons_prev_layer or
        # som_neuron_weight_vector_size]
        # Input Tensor Shape input_local_update_batch: [batch_size, (kernel_width*kernel_height), 1, 1,
        # num_som_neurons_prev_layer]
        # Input Tensor Shape neighborhood_coeff_batch: [batch_size, (1), num_maps, num_som_neurons]
        # Input Tensor Shape output_local_update_batch: [batch_size, kernel_width*kernel_height, num_maps,
        # num_som_neurons, num_som_neurons_prev_layer]
        # Output Tensor Shape: [num_maps, num_som_neurons, num_som_neurons_prev_layer or som_neuron_weight_vector_size]
        delta_local_weights_op = hebb(local_weights, modified_input_batch, neighborhood_coeff_batch,
                                   output_local_update_batch, local_learning_rate, reduce_sum_axis, reduce_mean_axis)

        # Output Tensor Shape: [num_maps, num_som_neurons, num_som_neurons_prev_layer or som_neuron_weight_vector_size]
        return delta_local_weights_op


def proposedLocalLearningOja(local_weights, input_local_update_batch, neighborhood_coeff_batch,
                             output_local_update_batch, local_learning_rate, reduce_sum_axis, reduce_mean_axis,
                             input_modification_coeff=1.0):
    """
    Calculates the delta weights following Our Rule (12) in the paper with Oja (d_w = a*y*(x-y*w),
    where x=x-r*sum_k(y*w))
    element-wise for the given input and output batch and weigths. The elemet-wise delta weights are located in each
    spatial entry of the output and are the mean over every sample in the batch.
    :param local_weights: (Tensor) The local weight to calculate the delta weights for.
                        Shape: [num_maps, num_som_neurons, num_som_neurons_prev_layer or som_neuron_weight_vector_size]
    :param input_local_update_batch: (Tensor) The input_batch for the Rule to calculate the local weights.
                                     Shape: [batch_size, (kernel_width*kernel_height), 1, 1,
                                     num_som_neurons_prev_layer or som_neuron_weight_vector_size]
    :param neighborhood_coeff_batch: (Tensor) The neighborhood coefficients for Oja to calculate the local weights.
                                     of the weights, restricting the update for particular weights if =! 1.
                                     Shape: [batch_size, (1), num_maps, num_som_neurons, 1] or [1]
    :param output_local_update_batch: (Tensor) The local output_batch for theRule to calculate the local weights.
                                      Shape: [batch_size, (kernel_width*kernel_height), num_maps, num_som_neurons,
                                      num_som_neurons_prev_layer or som_neuron_weight_vector_size]
    :param local_learning_rate: (Float) The learning rate for the Rule.
    :param reduce_sum_axis: (Integer) The axis to reduce in the input to support bot local weight types.
    :param reduce_mean_axis: (Array) The axis to reduce the batch for the delta weights to support bot
                             local weight types.
    :param input_modification_coeff: (Integer) The coeff for the input modification. 1.0 by default.
    :return: delta_local_weights_op: (Tensor) The operation to calculate the delta weights.
    """
    with tf.variable_scope("proposedLocalLearningOja"):
        # Input Tensor Shape local_weights: [num_maps, num_som_neurons, num_som_neurons_prev_layer or
        # som_neuron_weight_vector_size]
        # Input Tensor Shape input_local_update_batch: [batch_size, (kernel_width*kernel_height), 1, 1,
        # num_som_neurons_prev_layer]
        # Input Tensor Shape neighborhood_coeff_batch: [batch_size, (1), num_maps, num_som_neurons]
        # Input Tensor Shape output_local_update_batch: [batch_size, kernel_width*kernel_height, num_maps,
        # num_som_neurons, num_som_neurons_prev_layer]
        # Input Tensor Shape input_modification_coeff: [1]
        # Output Tensor Shape output_local_update_batch: [batch_size, kernel_width*kernel_height, num_maps,
        # 1, num_som_neurons_prev_layer]
        modified_input_batch = tf.subtract(input_local_update_batch, tf.multiply(input_modification_coeff,
                                           tf.reduce_sum(tf.multiply(output_local_update_batch, local_weights),
                                           axis=reduce_sum_axis, keepdims=True)))

        # Input Tensor Shape local_weights: [num_maps, num_som_neurons, num_som_neurons_prev_layer or
        # som_neuron_weight_vector_size]
        # Input Tensor Shape input_local_update_batch: [batch_size, (kernel_width*kernel_height), 1, 1,
        # num_som_neurons_prev_layer]
        # Input Tensor Shape neighborhood_coeff_batch: [batch_size, (1), num_maps, num_som_neurons]
        # Input Tensor Shape output_local_update_batch: [batch_size, kernel_width*kernel_height, num_maps,
        # num_som_neurons, num_som_neurons_prev_layer]
        # Output Tensor Shape: [num_maps, num_som_neurons, num_som_neurons_prev_layer or som_neuron_weight_vector_size]
        delta_local_weights_op = oja(local_weights, modified_input_batch, neighborhood_coeff_batch,
                                  output_local_update_batch, local_learning_rate, reduce_sum_axis, reduce_mean_axis)

        # Output Tensor Shape: [num_maps, num_som_neurons, num_som_neurons_prev_layer or som_neuron_weight_vector_size]
        return delta_local_weights_op


def generalizedHebbianElemMul(local_weights, input_local_update_batch, neighborhood_coeff_batch,
                              output_local_update_batch, local_learning_rate, reduce_sum_axis, reduce_mean_axis,
                              input_modification_coeff=1.0):
    """
    Calculates the delta weights following Our Rule (14) in the paper with Oja (d_w = a*y*(x-y*w),
    where x=x-r*sum_k<i(y*w))
    element-wise for the given input and output batch and weigths. The elemet-wise delta weights are located in each
    spatial entry of the output and are the mean over every sample in the batch.
    :param local_weights: (Tensor) The local weight to calculate the delta weights for.
                        Shape: [num_maps, num_som_neurons, num_som_neurons_prev_layer or som_neuron_weight_vector_size]
    :param input_local_update_batch: (Tensor) The input_batch for the Rule to calculate the local weights.
                                     Shape: [batch_size, (kernel_width*kernel_height), 1, 1,
                                     num_som_neurons_prev_layer or som_neuron_weight_vector_size]
    :param neighborhood_coeff_batch: (Tensor) The neighborhood coefficients for Oja to calculate the local weights.
                                     of the weights, restricting the update for particular weights if =! 1.
                                     Shape: [batch_size, (1), num_maps, num_som_neurons, 1] or [1]
    :param output_local_update_batch: (Tensor) The local output_batch for theRule to calculate the local weights.
                                      Shape: [batch_size, (kernel_width*kernel_height), num_maps, num_som_neurons,
                                      num_som_neurons_prev_layer or som_neuron_weight_vector_size]
    :param local_learning_rate: (Float) The learning rate for the Rule.
    :param reduce_sum_axis: (Integer) The axis to reduce in the input to support bot local weight types.
    :param reduce_mean_axis: (Array) The axis to reduce the batch for the delta weights to support bot
                             local weight types.
    :param input_modification_coeff: (Integer) The coeff for the input modification. 1.0 by default.
    :return: delta_local_weights_op: (Tensor) The operation to calculate the delta weights.
    """
    with tf.variable_scope("generalizedHebbianElemMul"):
        # Input Tensor Shape local_weights: [num_maps, num_som_neurons, num_som_neurons_prev_layer or
        # som_neuron_weight_vector_size]
        # Input Tensor Shape input_local_update_batch: [batch_size, (kernel_width*kernel_height), 1, 1,
        # num_som_neurons_prev_layer]
        # Input Tensor Shape neighborhood_coeff_batch: [batch_size, (1), num_maps, num_som_neurons]
        # Input Tensor Shape output_local_update_batch: [batch_size, kernel_width*kernel_height, num_maps,
        # num_som_neurons, num_som_neurons_prev_layer]
        # Input Tensor Shape input_modification_coeff: [1]
        # Output Tensor Shape output_local_update_batch: [batch_size, kernel_width*kernel_height, num_maps,
        # num_som_neurons, num_som_neurons_prev_layer]
        modified_input_batch = tf.subtract(input_local_update_batch, tf.multiply(input_modification_coeff,
                                           tf.cumsum(tf.multiply(output_local_update_batch, local_weights),
                                           axis=reduce_sum_axis, exclusive=True)))

        # Input Tensor Shape local_weights: [num_maps, num_som_neurons, num_som_neurons_prev_layer or
        # som_neuron_weight_vector_size]
        # Input Tensor Shape input_local_update_batch: [batch_size, (kernel_width*kernel_height), 1, 1,
        # num_som_neurons_prev_layer]
        # Input Tensor Shape neighborhood_coeff_batch: [batch_size, (1), num_maps, num_som_neurons]
        # Input Tensor Shape output_local_update_batch: [batch_size, kernel_width*kernel_height, num_maps,
        # num_som_neurons, num_som_neurons_prev_layer]
        # Output Tensor Shape: [num_maps, num_som_neurons, num_som_neurons_prev_layer or som_neuron_weight_vector_size]
        delta_local_weights_op = oja(local_weights, modified_input_batch, neighborhood_coeff_batch,
                                  output_local_update_batch, local_learning_rate, reduce_sum_axis, reduce_mean_axis)

        # Output Tensor Shape: [num_maps, num_som_neurons, num_som_neurons_prev_layer or som_neuron_weight_vector_size]
        return delta_local_weights_op

def generalizedHebbianElemMulNoWeight(local_weights, input_local_update_batch, neighborhood_coeff_batch,
                              output_local_update_batch, local_learning_rate, reduce_sum_axis, reduce_mean_axis,
                              input_modification_coeff=1.0):
    """
    Calculates the delta weights following Our Rule (13) in the paper with Oja (d_w = a*y*(x-y*w),
    where x=x-r*sum_k<i(y)
    element-wise for the given input and output batch and weigths. The elemet-wise delta weights are located in each
    spatial entry of the output and are the mean over every sample in the batch.
    :param local_weights: (Tensor) The local weight to calculate the delta weights for.
                        Shape: [num_maps, num_som_neurons, num_som_neurons_prev_layer or som_neuron_weight_vector_size]
    :param input_local_update_batch: (Tensor) The input_batch for the Rule to calculate the local weights.
                                     Shape: [batch_size, (kernel_width*kernel_height), 1, 1,
                                     num_som_neurons_prev_layer or som_neuron_weight_vector_size]
    :param neighborhood_coeff_batch: (Tensor) The neighborhood coefficients for Oja to calculate the local weights.
                                     of the weights, restricting the update for particular weights if =! 1.
                                     Shape: [batch_size, (1), num_maps, num_som_neurons, 1] or [1]
    :param output_local_update_batch: (Tensor) The local output_batch for theRule to calculate the local weights.
                                      Shape: [batch_size, (kernel_width*kernel_height), num_maps, num_som_neurons,
                                      num_som_neurons_prev_layer or som_neuron_weight_vector_size]
    :param local_learning_rate: (Float) The learning rate for the Rule.
    :param reduce_sum_axis: (Integer) The axis to reduce in the input to support bot local weight types.
    :param reduce_mean_axis: (Array) The axis to reduce the batch for the delta weights to support bot
                             local weight types.
    :param input_modification_coeff: (Integer) The coeff for the input modification. 1.0 by default.
    :return: delta_local_weights_op: (Tensor) The operation to calculate the delta weights.
    """
    with tf.variable_scope("generalizedHebbianElemMulNoWeight"):
        # Input Tensor Shape local_weights: [num_maps, num_som_neurons, num_som_neurons_prev_layer or
        # som_neuron_weight_vector_size]
        # Input Tensor Shape input_local_update_batch: [batch_size, (kernel_width*kernel_height), 1, 1,
        # num_som_neurons_prev_layer]
        # Input Tensor Shape neighborhood_coeff_batch: [batch_size, (1), num_maps, num_som_neurons]
        # Input Tensor Shape output_local_update_batch: [batch_size, kernel_width*kernel_height, num_maps,
        # num_som_neurons, num_som_neurons_prev_layer]
        # Input Tensor Shape input_modification_coeff: [1]
        # Output Tensor Shape output_local_update_batch: [batch_size, kernel_width*kernel_height, num_maps,
        # num_som_neurons, num_som_neurons_prev_layer]
        modified_input_batch = tf.subtract(input_local_update_batch, tf.multiply(input_modification_coeff,
                                           tf.cumsum(output_local_update_batch, axis=reduce_sum_axis, exclusive=True)))

        # Input Tensor Shape local_weights: [num_maps, num_som_neurons, num_som_neurons_prev_layer or
        # som_neuron_weight_vector_size]
        # Input Tensor Shape input_local_update_batch: [batch_size, (kernel_width*kernel_height), 1, 1,
        # num_som_neurons_prev_layer]
        # Input Tensor Shape neighborhood_coeff_batch: [batch_size, (1), num_maps, num_som_neurons]
        # Input Tensor Shape output_local_update_batch: [batch_size, kernel_width*kernel_height, num_maps,
        # num_som_neurons, num_som_neurons_prev_layer]
        # Output Tensor Shape: [num_maps, num_som_neurons, num_som_neurons_prev_layer or som_neuron_weight_vector_size]
        delta_local_weights_op = oja(local_weights, modified_input_batch, neighborhood_coeff_batch,
                                  output_local_update_batch, local_learning_rate, reduce_sum_axis, reduce_mean_axis)

        # Output Tensor Shape: [num_maps, num_som_neurons, num_som_neurons_prev_layer or som_neuron_weight_vector_size]
        return delta_local_weights_op


def noise(local_weights, input_local_update_batch, neighborhood_coeff_batch, output_local_update_batch,
                       local_learning_rate, reduce_sum_axis, reduce_mean_axis):
    """
    Calculates the delta weights randomly element-wise for the given input and output batch and weigths.
    The elemet-wise delta weights are located in each spatial entry of the output.
    :param local_weights: (Tensor) The local weight to calculate the delta weights for.
                        Shape: [num_maps, num_som_neurons, num_som_neurons_prev_layer or som_neuron_weight_vector_size]
    :param input_local_update_batch: (Tensor) The input_batch for the Rule to calculate the local weights.
                                     Shape: [batch_size, (kernel_width*kernel_height), 1, 1,
                                     num_som_neurons_prev_layer or som_neuron_weight_vector_size]. Not used here!
    :param neighborhood_coeff_batch: (Tensor) The neighborhood coefficients for Oja to calculate the local weights.
                                     of the weights, restricting the update for particular weights if =! 1.
                                     Shape: [batch_size, (1), num_maps, num_som_neurons, 1] or [1]. Not used here!
    :param output_local_update_batch: (Tensor) The local output_batch for theRule to calculate the local weights.
                                      Shape: [batch_size, (kernel_width*kernel_height), num_maps, num_som_neurons,
                                      num_som_neurons_prev_layer or som_neuron_weight_vector_size]. Not used here!
    :param local_learning_rate: (Float) The learning rate for the Rule. Not used here!
    :param reduce_sum_axis: (Integer) The axis to reduce in the input to support bot local weight types. Not used here!
    :param reduce_mean_axis: (Array) The axis to reduce the batch for the delta weights to support bot
                             local weight types. Not used here!
    :return: delta_local_weights_op: (Tensor) The operation to calculate the delta weights. Not used here!
    """
    with tf.variable_scope("noise"):
        # Output Tensor Shape local_weights: [num_maps, num_som_neurons, num_som_neurons_prev_layer or
        # som_neuron_weight_vector_size]
        m, n, d = local_weights.get_shape()
        delta_local_weights_op = tf.random_uniform([m.value, n.value, d.value], minval=-1.0, maxval=1.0,
                                                   dtype=tf.float32)

        # Output Tensor Shape: [num_maps, num_som_neurons, num_som_neurons_prev_layer or som_neuron_weight_vector_size]
        return delta_local_weights_op

def static(local_weights, input_local_update_batch, neighborhood_coeff_batch, output_local_update_batch,
                       local_learning_rate, reduce_sum_axis, reduce_mean_axis):
    """
    Stets the delta weights to zero.
    The elemet-wise delta weights are located in each spatial entry of the output.
    :param local_weights: (Tensor) The local weight to calculate the delta weights for.
                        Shape: [num_maps, num_som_neurons, num_som_neurons_prev_layer or som_neuron_weight_vector_size]
    :param input_local_update_batch: (Tensor) The input_batch for the Rule to calculate the local weights.
                                     Shape: [batch_size, (kernel_width*kernel_height), 1, 1,
                                     num_som_neurons_prev_layer or som_neuron_weight_vector_size]. Not used here!
    :param neighborhood_coeff_batch: (Tensor) The neighborhood coefficients for Oja to calculate the local weights.
                                     of the weights, restricting the update for particular weights if =! 1.
                                     Shape: [batch_size, (1), num_maps, num_som_neurons, 1] or [1]. Not used here!
    :param output_local_update_batch: (Tensor) The local output_batch for theRule to calculate the local weights.
                                      Shape: [batch_size, (kernel_width*kernel_height), num_maps, num_som_neurons,
                                      num_som_neurons_prev_layer or som_neuron_weight_vector_size]. Not used here!
    :param local_learning_rate: (Float) The learning rate for the Rule. Not used here!
    :param reduce_sum_axis: (Integer) The axis to reduce in the input to support bot local weight types. Not used here!
    :param reduce_mean_axis: (Array) The axis to reduce the batch for the delta weights to support bot
                             local weight types. Not used here!
    :return: delta_local_weights_op: (Tensor) The operation to calculate the delta weights. Not used here!
    """
    with tf.variable_scope("static"):
        # Output Tensor Shape local_weights: [num_maps, num_som_neurons, num_som_neurons_prev_layer or
        # som_neuron_weight_vector_size]
        m, n, d = local_weights.get_shape()
        delta_local_weights_op = tf.zeros([m.value, n.value, d.value], dtype=tf.float32)

        # Output Tensor Shape: [num_maps, num_som_neurons, num_som_neurons_prev_layer or som_neuron_weight_vector_size]
        return delta_local_weights_op


def generalizedHebbianElemMul05(local_weights, input_local_update_batch, neighborhood_coeff_batch,
                              output_local_update_batch,
                              local_learning_rate, reduce_sum_axis, reduce_mean_axis):
    return generalizedHebbianElemMul(local_weights, input_local_update_batch, neighborhood_coeff_batch,
                              output_local_update_batch, local_learning_rate, reduce_sum_axis, reduce_mean_axis,
                              input_modification_coeff=0.5)

def proposedLocalLearningHebb05(local_weights, input_local_update_batch, neighborhood_coeff_batch,
                                output_local_update_batch, local_learning_rate, reduce_sum_axis, reduce_mean_axis):
    return proposedLocalLearningHebb(local_weights, input_local_update_batch, neighborhood_coeff_batch,
                                     output_local_update_batch, local_learning_rate, reduce_sum_axis, reduce_mean_axis,
                                     input_modification_coeff=0.5)

def generalizedHebbianElemMulNoWeight05(local_weights, input_local_update_batch, neighborhood_coeff_batch,
                              output_local_update_batch,
                              local_learning_rate, reduce_sum_axis, reduce_mean_axis):
    return generalizedHebbianElemMulNoWeight(local_weights, input_local_update_batch, neighborhood_coeff_batch,
                              output_local_update_batch, local_learning_rate, reduce_sum_axis, reduce_mean_axis,
                              input_modification_coeff=0.5)
