import tensorflow as tf
import numpy as np

from csnnLib import bmuMetrics
from csnnLib import learningRules
from csnnLib import neighborhoodFunctions
from csnnLib import evaluation

local_weights_types = ["betweenNeurons", "betweenInputAndNeurons", 0]

def localNeuronWeights(input_batch, num_maps, num_som_neurons_per_map, num_som_neurons_prev_layer, patch_size, name):
    """
    Creates local weights between the neurons (features) in the input_batch and the current layer neurons.
    :param input_batch: (Tensor) The input_batch to calculate the distance with.
                        Shape: [batch_size, 1 or num_maps, 1 or num_som_neurons, weight_vector_size]
    :param num_maps: (Integer) The number of SOM maps of the current layer.
    :param num_som_neurons_per_map: (Integer) The number of SOM neurons per map of the current layer.
    :param num_som_neurons_prev_layer: (Integer) The number of all SOM neurons of the previous layer.
    :param patch_size: (Integer): The size of the patch for the current layer.
    :param name: (String) The name of this local weights for the Tensorflow scope.
    :return: output_batch_re: (Tensor) The output of the local weights layer for the distance metric.
                              Shape: [batch_size, num_maps, num_som_neuron, patch_size]
    :return: output_local_update_batch: (Tensor) The output of the local weights layer for the local learning.
                              Shape: [batch_size, kernel_width*kernel_height, num_maps,
                              num_som_neuron, num_som_neurons_prev_layer]
    :return: input_local_update_batch: (Tensor) The input of the local weights layer for the local learning.
                              Shape: [batch_size, kernel_width*kernel_height, 1, 1, num_som_neurons_prev_layer]
    :return: local_weights: (Tensor) The local weights. Shape: [num_maps, num_som_neuron, num_som_neurons_prev_layer]
    """
    with tf.variable_scope('localWeights'+str(name)):
        # Define fc weights between this layers som and the input or the prv layer som.
        # Tensor Shape: [num_maps, num_som_neuron, num_som_neurons_prev_layer]
        local_weights = tf.get_variable(name + "Weights",
                                        [num_maps, num_som_neurons_per_map, num_som_neurons_prev_layer],
                                        initializer=tf.random_uniform_initializer(-1.0, 1.0))

        # Output Tensor Shape: [batch_size, kernel_width*kernel_height, num_som_neurons_prev_layer]
        input_local_update_batch = tf.reshape(input_batch, [-1, patch_size // num_som_neurons_prev_layer,
                                                      num_som_neurons_prev_layer])

        # Output Tensor Shape: [batch_size, kernel_width*kernel_height, 1, 1, num_som_neurons_prev_layer]
        input_local_update_batch = tf.expand_dims(tf.expand_dims(input_local_update_batch, 2), 2)

        # Output Tensor Shape: [batch_size, kernel_width*kernel_height, num_maps, num_som_neuron,
        # num_som_neurons_prev_layer]
        output_local_update_batch = tf.multiply(input_local_update_batch, tf.expand_dims(tf.expand_dims(local_weights,
                                                                                                    axis=0), axis=0))

        # Output Tensor Shape: [batch_size, num_maps, num_som_neuron, num_som_neurons_prev_layer,
        # kernel_width*kernel_height]
        output_batch_re = tf.transpose(output_local_update_batch, [0, 2, 3, 4, 1])

        # Output Tensor Shape: [batch_size, num_maps, num_som_neuron,
        # patch_size=num_som_neurons_prev_layer*kernel_width*kernel_height]
        output_batch_re = tf.reshape(output_batch_re, [-1, num_maps, num_som_neurons_per_map, patch_size])

        # Output Tensor Shape output_batch_re: [batch_size, num_maps, num_som_neuron, patch_size]
        # Output Tensor Shape output_local_update_batch: [batch_size, kernel_width*kernel_height, num_maps,
        # num_som_neuron, num_som_neurons_prev_layer]
        # Output Tensor Shape input_local_update_batch: [batch_size, kernel_width*kernel_height, 1, 1,
        # num_som_neurons_prev_layer]
        # Output Tensor Shape local_weights: [num_maps, num_som_neuron, num_som_neurons_prev_layer]
        return output_batch_re, output_local_update_batch, input_local_update_batch, local_weights


def localInputWeights(input_batch, num_maps, num_som_neurons_per_map, patch_size, name):
    """
    Creates local weights between the input patches in the input_batch and the current layer neurons.
    :param input_batch: (Tensor) The input_batch to calculate the distance with.
                        Shape: [batch_size, 1 or num_maps, 1 or num_som_neurons, weight_vector_size]
    :param num_maps: (Integer) The number of SOM maps of the current layer.
    :param num_som_neurons_per_map: (Integer) The number of SOM neurons per map of the current layer.
    :param num_som_neurons_prev_layer: (Integer) The number of all SOM neurons of the previous layer.
    :param patch_size: (Integer): The size of the patch for the current layer.
    :param name: (String) The name of this local weights for the Tensorflow scope.
    :return: output_local_update_batch: (Tensor) The output of the local weights layer for the SOM and local learning.
                              Shape: [batch_size, num_maps, num_som_neuron, patch_size]
    :return: input_local_update_batch: (Tensor) The input of the local weights layer for the local learning.
                              Shape: [batch_size, 1, 1, patch_size]
    :return: local_weights: (Tensor) The local weights. Shape: [num_maps, num_som_neuron, patch_size]
    """
    with tf.variable_scope('localInputWeights' + str(name)):
        # Define fc weights between this layers som and the input or the prv layer som.
        # Tensor Shape: [num_maps, num_som_neuron, patch_size]
        local_weights = tf.get_variable(name + "Weights",
                                        [num_maps, num_som_neurons_per_map, patch_size],
                                        initializer=tf.random_uniform_initializer(-1.0, 1.0))

        # Output Tensor Shape input_local_update_batch: [batch_size, 1, 1, patch_size]
        input_local_update_batch = tf.expand_dims(tf.expand_dims(input_batch, 1), 1)

        # Output Tensor Shape output_local_update_batch: [batch_size, num_maps, num_som_neuron, patch_size]
        output_local_update_batch = tf.multiply(input_local_update_batch, tf.expand_dims(local_weights, axis=0))

        # Output Tensor Shape output_local_update_batch: [batch_size, num_maps, num_som_neuron, patch_size]
        # Output Tensor Shape input_local_update_batch: [batch_size, 1, 1, patch_size]
        # Output Tensor Shape local_weights: [num_maps, num_som_neuron, patch_size]
        return output_local_update_batch, input_local_update_batch, local_weights


def createLocalWeights(local_weights_type, input_batch, num_maps, num_som_neurons_per_map, som_kernel_depth,
                       patch_size, name):
    """
    Creates local weights for the CSNN layer.
    :param local_weights_type: (String) The type of local weights: "betweenNeurons" or "betweenInputAndNeurons".
    :param input_batch: (Tensor) The input batch containing all patches. Shape: [batch_size, patch_size]
    :param num_maps: (Integer) The number of SOM maps of the current layer.
    :param num_som_neurons_per_map: (Integer) The number of SOM neurons per map of the current layer.
    :param som_kernel_depth: (Integer) The depth of the kernel (the number of all SOM neurons of the previous layer).
    :param patch_size: (Integer): The size of the patch for the current layer.
    :param name: (String) The name of this local weights for the Tensorflow scope.
    :return: input_batch_metric: (Tensor) The input for the distance metric of the SOM.
                                 Shape: [batch_size, num_maps or 1, num_som_neuron or 1, patch_size]
    :return: output_local_update_batch: (Tensor) The output of the local weights layer for the SOM and local learning.
                              Shape: [batch_size, (kernel_width*kernel_height), num_maps, num_som_neuron,
                              num_som_neurons_prev_layer or patch_size] or None.
    :return: input_local_update_batch: (Tensor) The input of the local weights layer for the local learning.
                                       Shape: [batch_size, (kernel_width*kernel_height), 1, 1,
                                       patch_size or num_som_neurons_prev_layer] or None.
    :return: local_weights: (Tensor) The local weights.
                             Shape: [num_maps, num_som_neuron, patch_size or num_som_neurons_prev_layer] or None.
    """
    if local_weights_type == "betweenNeurons":
        # Input Tensor Shape input_batch: [batch_size, patch_size]
        # Output Tensor Shape input_batch_metric: [batch_size, num_maps, num_som_neuron, patch_size]
        # Output Tensor Shape output_local_update_batch: [batch_size, kernel_width*kernel_height, num_maps,
        # num_som_neuron, num_som_neurons_prev_layer]
        # Output Tensor Shape input_local_update_batch: [batch_size, kernel_width*kernel_height, 1, 1,
        # num_som_neurons_prev_layer]
        # Output Tensor Shape local_weights: [num_maps, num_som_neuron, num_som_neurons_prev_layer]
        input_batch_metric, output_local_update_batch, input_local_update_batch, local_weights = \
           localNeuronWeights(input_batch, num_maps, num_som_neurons_per_map, som_kernel_depth, patch_size,
                              name+"Local")
    elif local_weights_type == "betweenInputAndNeurons":
        # Input Tensor Shape input_batch: [batch_size, patch_size]
        # Output Tensor Shape input_batch_metric: [batch_size, num_maps, num_som_neuron,
        # som_neuron_weight_vector_size]
        # Output Tensor Shape output_local_update_batch: [batch_size, num_maps, num_som_neuron, patch_size]
        # Output Tensor Shape input_local_update_batch: [batch_size, 1, 1, patch_size]
        # Output Tensor Shape local_weights: [num_maps, num_som_neuron, patch_size]
        output_local_update_batch, input_local_update_batch, local_weights = \
            localInputWeights(input_batch, num_maps, num_som_neurons_per_map, patch_size,
                              name+"Local")
        input_batch_metric = output_local_update_batch
    else:
        output_local_update_batch, input_local_update_batch, local_weights = [None, None, None]
        # Input Tensor Shape input_batch: [batch_size, patch_size]
        # Output Tensor Shape input_batch_metric: [batch_size, 1, 1, patch_size]
        input_batch_metric = tf.expand_dims(tf.expand_dims(input_batch, axis=1), axis=1)

    # Output Tensor Shape input_batch_metric: [batch_size, num_maps or 1, num_som_neuron or 1, patch_size]
    # Output Tensor Shape output_local_update_batch: [batch_size, (kernel_width*kernel_height), num_maps,
    # num_som_neuron, num_som_neurons_prev_layer or patch_size] or None
    # Output Tensor Shape input_local_update_batch: [batch_size, (kernel_width*kernel_height), 1, 1,
    # num_som_neurons_prev_layer or patch_size] or None
    # Output Tensor Shape local_weights: [num_maps, num_som_neuron, num_som_neurons_prev_layer or patch_size] or None
    return input_batch_metric, output_local_update_batch, input_local_update_batch, local_weights


def createInputPatches(input_batch, som_kernel, som_kernel_depth, strides, padding, name, verbose=False):
    """
    Creates the convolution patches, with the given stride and padding analog to the convolution patches in a CNN.
    :param input_batch: (Tensor) The input batch. Shape: [batch_size, input_height, input_width, input_depth]
    :param som_kernel: (Array) The kernel of the SOM analog to the CNN kernel: [k_h, k_w].
    :param som_kernel_depth: (Integer) The depth of the SOM kernel similar to the input depth.
    :param strides:  (Array) The strides of the SOM analog to the CNN strides: [s_h, s_w].
    :param padding:  (Array) The padding of the SOM analog to the CNN paddings: "SAME" or "VALID".
    :param name: (String) The name of this operation for the Tensorflow scope.
    :param verbose: (Boolean) If true, patches will be shown in Tensorboard. False by default.
    :return: input_patches_batch: (Tensor) The input_batch containing all the patches on theri position for
                                  visualization. Shape: [batch_size, num_conv_patches_h, num_conv_patches_w,
                                  patch_size=patch_size*input_depth]
    :return: input_patches_batch_re: (Tensor) The input_batch containing all the patches in other form for local
                                     weights and SOM.
                                     Shape:  [batch_size=batch_size*num_conv_patches_h*num_conv_patches_w,
                                     patch_size=som_kernel[0]*som_kernel[1]*som_kernel_depth]
    :return: num_conv_patches_h: (Integer) The number of conv_patches in the height dimension.
    :return: num_conv_patches_w: (Integer) The number of conv_patches in the width dimension.
    :return: patch_size: (Integer) The patch size=som_kernel[0] * som_kernel[1] * som_kernel_depth.

    """
    # Create the input patches
    with tf.variable_scope('createAndShapePatches' + str(name)):
        # Calculate the size of the weight bector of one som neuron.
        patch_size = som_kernel[0] * som_kernel[1] * som_kernel_depth

        # Reshaping the kernel and strides for extract_image_patches.
        k_sizes = [1, som_kernel[0], som_kernel[1], 1]
        strides = [1, strides[0], strides[1], 1]

        # Extracting the image_patches. "Computing the convolution patches".
        # Input Tensor Shape input_batch: [batch_size, input_height, input_width, input_depth]
        # Input Tensor Shape k_sizes: [1, som_kernel_height, som_kernel_width, 1]
        # Input Tensor Shape strides: [1, stride_h, stride_w, 1]
        # Input Tensor Shape padding: The type of the padding: valid or same.
        # Output Tensor Shape: [batch_size, num_conv_patches_h, num_conv_patches_w, patch_size*input_depth]
        input_patches_batch = tf.extract_image_patches(images=input_batch, ksizes=k_sizes, strides=strides,
                                                       rates=[1, 1, 1, 1], padding=padding)

        # Saving the num_conv_patches_h and num_conv_patches_w for a reshape later on.
        b, num_conv_patches_h, num_conv_patches_w, _ = input_patches_batch.get_shape()
        num_conv_patches_h = num_conv_patches_h.value
        num_conv_patches_w = num_conv_patches_w.value

        # Reshaping the input_patches_batch: All input_patches go in the batch_size dimension!
        # We continue to refer to the the batch size dimension as batch size dimension.
        # Now the som can compute its output for each patch and batch without the use of a for loop.
        # Input Tensor Shape: [batch_size, num_conv_patches_h, num_conv_patches_w, patch_size*input_depth]
        # Output Tensor Shape: [batch_size=batch_size*num_conv_patches_h*num_conv_patches_w,
        # patch_size=som_kernel[0]*som_kernel[1]*som_kernel_depth]
        input_patches_batch_re = tf.reshape(input_patches_batch, [-1, patch_size])

        if verbose:
            input_patches_vis_batch = tf.reshape(input_patches_batch_re, [-1, som_kernel[0], som_kernel[1],
                                                                          som_kernel_depth])
            tf.summary.image("Patches", input_patches_vis_batch, max_outputs=10)

        # Output Tensor Shape: [batch_size, num_conv_patches_h, num_conv_patches_w, patch_size=patch_size*input_depth]
        # Output Tensor Shape: [batch_size=batch_size*num_conv_patches_h*num_conv_patches_w,
        # patch_size=som_kernel[0]*som_kernel[1]*som_kernel_depth]
        # Output Tensor Shape: [1]
        # Output Tensor Shape: [1]
        return input_patches_batch, input_patches_batch_re, num_conv_patches_h, num_conv_patches_w, patch_size


def __onlyBestBmuUpdate(distances_batch, neighborhood_coeff_batch):
    """
    Sets neighborhood_coeffs of all maps to zero, but not for the map with the maximal BMU. WARNING:
    Does not support distance metrics for BMUs with min distance!
    :param distances_batch: (Tensor) The distances of the SOM neurons to the CSNN layer patches.
                            Shape: [batch_size, num_maps, num_som_neurons]
    :param neighborhood_coeff_batch: (Tensor) The neighborhood_coeffs of the SOM neurons.
                                     Shape: [batch_size, num_maps, num_som_neurons]
    :return: neighborhood_coeff_batch_mask: (Tensor) The neighborhood_coeffs of the SOM neurons.
                                            Shape: [batch_size, num_maps, num_som_neurons]
    """
    # Input Tensor Shape: [batch_size, num_maps, num_som_neurons]
    # Output Tensor Shape: [batch_size]
    max_map_index_batch = tf.math.argmax(tf.reduce_max(distances_batch, axis=2), axis=1)

    # Input Tensor Shape: [batch_size]
    # Output Tensor Shape: [batch_size, num_maps]
    maps = neighborhood_coeff_batch.get_shape()[1].value
    mask_batch = tf.one_hot(max_map_index_batch, maps)

    # Input Tensor Shape: [batch_size, num_maps]
    # Output Tensor Shape: [batch_size, num_maps, 1]
    mask_batch = tf.expand_dims(mask_batch, axis=2)

    # Input Tensor Shape neighborhood_coeff_batch: [batch_size, num_maps]
    # Input Tensor Shape mask_batch: [batch_size, num_maps, 1]
    # Output Tensor Shape: [batch_size, num_maps, num_som_neurons]
    neighborhood_coeff_batch_mask = tf.multiply(neighborhood_coeff_batch, mask_batch)

    # Output Tensor Shape: [batch_size, num_maps, num_som_neurons]
    return neighborhood_coeff_batch_mask


def __neuronCords(width, height):
    """
    Returns an array with the 2D SOM-neuron coordinates given the width and height of the SOM map.
    :param width: (Integer) The width of the SOM map.
    :param height: (Integer) The height of the SOM map.
    :return: cords: (Array) The 2D SOM-neuron coordinates: [[0,0], ....]
    """
    for i in range(width):
        for j in range(height):
            yield np.array([i, j])


def __csnnLayerTrain(neuron_cords, neighborhood_function, bmus_batch, train_interval, distances_batch,
                     som_weights, local_weights, output_local_update_batch, input_local_update_batch,
                     som_learning_rule, local_learning_rule, local_weights_type, local_neighborhood_coeff_type,
                     som_learning_rate, local_learning_rate, input_patches_batch_re, **neighborhoodFunction_args):
    """
    The training graph of the CSNN layer only executed when the current Layer is in training.
    Defines the operations for learning.
    :param neuron_cords: (Tensor) The 2D SOM-neuron coordinates of this layer. Shape: [num_som_neurons_per_map, 2]
    :param neighborhood_function: (String) The type of neighborhood function to use. E.g. "gaussian".
    :param bmus_batch: (Tensor) The batch containing the BMUs. Shape: [batch_size ,num_maps]
    :param train_interval: (Array) The training inverval of the layer. E.g. [0, 10000].
    :param distances_batch: (Tensor) The batch containing the distances of the SOM-neurons to the input.
                            Shape: [batch_size, num_maps, num_som_neurons]
    :param som_weights: (Tensor) The som weights of this layer. Shape: [num_maps, num_som_neurons_per_map,
                         patch_size=som_neuron_weight_vector_size]
    :param local_weights: (Tensor) The local weights of this layer. Shape: [num_maps, num_som_neuron,
                          num_som_neurons_prev_layer or patch_size] or None
    :param output_local_update_batch: (Tensor) The output of the local weights layer for the SOM and local learning.
                                       Shape: [batch_size, (kernel_width*kernel_height), num_maps, num_som_neuron,
                                       num_som_neurons_prev_layer or patch_size] or None.
    :param input_local_update_batch: (Tensor) The input of the local weights layer for the local learning.
                                      Shape: [batch_size, (kernel_width*kernel_height), 1, 1,
                                      patch_size or num_som_neurons_prev_layer] or None.
    :param som_learning_rule: (String) The som learning rule to use. E.g. "convSomLearning".
    :param local_learning_rule: (String) The local learning rule to use.
    :param local_weights_type: (String) The type of local weights: "betweenNeurons" or "betweenInputAndNeurons".
    :param local_neighborhood_coeff_type: (String) The type of neighborhood_coeff to use for the local update.
                                          See: learningRules.__prepareForLocalWeightsLearning
    :param som_learning_rate: (Float) The learning rate of the som weights.
    :param local_learning_rate: (Float) The learning rate of the local weights.
    :param input_patches_batch_re: (Tensor) The input of the lSOM for the som learning.
                                       Shape: [batch_size=batch_size*num_conv_patches_h*num_conv_patches_w,
                                       patch_size=som_kernel[0]*som_kernel[1]*som_kernel_depth]
    :param neighborhoodFunction_args: (**) The parameters for the neighborhood function.
    :return: delta_som_weights: (Tensor) The delta weights for the som(s) of this layer.
                                 Shape: [num_maps, num_som_neurons, som_neuron_weight_vector_size]
    :return: delta_local_weights: (Tensor) The delta local weights of this layer.
                                 Shape: [num_maps, num_som_neurons, num_som_neurons_prev_layer
                                 or som_neuron_weight_vector_size]
    :return: neighborhood_coeff_batch: (Tensor) The neighborhood_coeffs for each neuron, map and batch.
                                      Shape: [batch_size, num_maps, num_som_neurons_per_map]
    """
    # Split the hyperparameter, One is for the neighborhood_function type, the other to check if the BMUs of all maps
    # should be updated.
    neighborhood_function_params = neighborhood_function.split("_")
    neighborhood_function = neighborhood_function_params[0]

    # Compute the neighborhood coeffs for each neuron.
    # Input Tensor Shape neuron_cords: [num_som_neurons_per_map, 2]
    # Input Tensor Shape bmus: [batch_size ,num_maps]
    # Output Tensor Shape: [batch_size, num_maps, num_som_neurons_per_map] -
    # the neighborhood_coeff for each neuron, map and batch
    neighborhood_coeff_batch = getattr(neighborhoodFunctions, neighborhood_function) \
        (neuron_cords, bmus_batch, train_interval[1] - train_interval[0], **neighborhoodFunction_args)

    # Just use the neighborhood coeffs of the map with the best BMU.
    # Input Tensor Shape distances_batch: [batch_size, num_maps, num_som_neurons]
    # Input Tensor Shape neighborhood_coeff_batch: [batch_size, num_maps, num_som_neurons]
    # Output Tensor Shape: [batch_size, num_maps, num_som_neurons]
    if not "allBmus" in neighborhood_function_params:
        neighborhood_coeff_batch = __onlyBestBmuUpdate(distances_batch, neighborhood_coeff_batch)

    # Output Tensor Shape delta_local_weights: [num_maps, num_som_neurons, num_som_neurons_prev_layer
    # or som_neuron_weight_vector_size]
    if local_weights_type:
        neighborhood_coeff_local_batch, reduce_sum_axis, reduce_mean_axis, = \
            learningRules.__prepareForLocalWeightsLearning(neighborhood_coeff_batch, local_neighborhood_coeff_type,
                                                           local_weights_type)
        delta_local_weights = getattr(learningRules, local_learning_rule)(local_weights, input_local_update_batch,
                                            neighborhood_coeff_local_batch, output_local_update_batch,
                                                    local_learning_rate, reduce_sum_axis, reduce_mean_axis)
    else:
        delta_local_weights = tf.zeros(som_weights.get_shape())

    # Compute the new weights for this csnn layer.
    # Input Tensor Shape learn_rate: [1]
    # Input Tensor Shape neighborhood_coeff: [batch_size*num_conv_patches_h*num_conv_patches_w, num_maps,
    # num_som_neurons_per_map]
    # Input Tensor Shape input_patches_re: [batch_size*num_conv_patches_h*num_conv_patches_w, 1, 1,
    # som_neuron_weight_vector_size]
    # Input Tensor Shape som_net_weights_per_map: [num_maps, num_som_neurons_per_map, som_neuron_weight_vector_size]
    # Output Tensor Shape delta_som_weights: [num_maps, num_som_neurons, som_neuron_weight_vector_size]
    delta_som_weights = getattr(learningRules, som_learning_rule)(som_learning_rate, neighborhood_coeff_batch,
                                                      tf.expand_dims(tf.expand_dims(input_patches_batch_re, axis=1),
                                                                     axis=1), som_weights)

    # Output Tensor Shape delta_som_weights: [num_maps, num_som_neurons, som_neuron_weight_vector_size]
    # Output Tensor Shape delta_local_weights: [num_maps, num_som_neurons, num_som_neurons_prev_layer
    # or som_neuron_weight_vector_size]
    # Output Tensor Shape neighborhood_coeff_batch: [batch_size, num_maps, num_som_neurons_per_map] -
    # the neighborhood_coeffs for each neuron, map and batch
    return delta_som_weights, delta_local_weights, neighborhood_coeff_batch


def __csnnLayerInfer(som_net_weights, local_weights, local_weights_type, distances_batch):
    """
    The inference graph of the CSNN layer only executed when the current Layer is not training.
    Defines the operations for learning.
    :param som_weights: (Tensor) The som weights of this layer. Shape: [num_maps, num_som_neurons_per_map,
                        patch_size=som_neuron_weight_vector_size]
    :param local_weights: (Tensor) The local weights of this layer. Shape: [num_maps, num_som_neuron,
                            num_som_neurons_prev_layer or patch_size] or None.
    :param local_weights_type: (String) The type of local weights: "betweenNeurons" or "betweenInputAndNeurons".
    :param distances_batch: (Tensor) The batch containing the distances of the SOM-neurons to the input.
                            Shape: [batch_size, num_maps, num_som_neurons]
    :return: delta_som_weights: (Tensor) The empty delta weights for the som(s) of this layer.
                                 Shape: [num_maps, num_som_neurons, som_neuron_weight_vector_size]
    :return: delta_local_weights: (Tensor) The empty delta local weights of this layer.
                                 Shape: [num_maps, num_som_neurons, num_som_neurons_prev_layer
                                 or som_neuron_weight_vector_size]
    :return: neighborhood_coeff_batch: (Tensor) The empty neighborhood_coeffs for each neuron, map and batch.
                                      Shape: [batch_size, num_maps, num_som_neurons_per_map]
    """
    delta_local_weights = tf.zeros(som_net_weights.get_shape())
    if (local_weights_type == "betweenNeurons") or (local_weights_type == "betweenInputAndNeurons"):
        delta_local_weights = tf.zeros(local_weights.get_shape())
    delta_som_weights = tf.zeros(som_net_weights.get_shape())
    m, n = som_net_weights.get_shape()[0:2]
    b = tf.shape(distances_batch)[0]
    neighborhood_coeff_batch = tf.zeros([b, m.value, n.value, 1])

    return delta_som_weights, delta_local_weights, neighborhood_coeff_batch


def csnnLayer(input_batch, som_grid, som_kernel, strides, train_interval=None, padding="SAME",
              bmu_metrics="convolutionDistance", local_weights_type="betweenNeurons", local_neighborhood_coeff_type=0,
              som_learning_rule="convSomLearning", local_learning_rule="proposedLocalLearning", som_learning_rate=0.1,
              local_learning_rate=0.005, neighborhood_function="gaussian",
              name="som2DGridLayer", verbose=False, **neighborhoodFunction_args):
    """
    Creates a CSNN layer which is described in our paper with its local weights, distance metrics, etc.
    This layer is very similar to a CNN layer.
    :param input_batch: (Tensor) The input to the CSNN layer.
                        Shape: [batch_size, input_height, input_width, input_depth].
    :param som_grid: (Array) The SOM-grid of the layer, which defines the number of features. E.g [10,10,3] creates
                     a layer with 3 maps of 10x10 Som-grids for 300 features.
    :param som_kernel: (Array) The kernel of the SOM analog to the CNN kernel: [k_h, k_w].
    :param strides: (Array) The strides of the SOM analog to the CNN strides: [s_h, s_w].
    :param train_interval: (Array) The training inverval of the layer. E.g. [0, 10000]. None by default.
    :param padding: (Array) The padding of the SOM analog to the CNN paddings: "SAME" or "VALID". "SAME" by default.
    :param bmu_metrics: (String) The distance metric to use in the SOM. "convolutionDistance" by default.
    :param local_weights_type: (String) The type of local weights: "betweenNeurons" or "betweenInputAndNeurons".
    :param local_neighborhood_coeff_type: (String) The type of neighborhood_coeff to use for the local update.
                                          See: learningRules.__prepareForLocalWeightsLearning
    :param som_learning_rule: (String) The som learning rule to use. "convSomLearning" by default.
    :param local_learning_rule: (String) The local learning rule to use. "proposedLocalLearning" by default.
    :param som_learning_rate: (Float) The learning rate of the som weights. 0.1 by default.
    :param local_learning_rate: (Float) The learning rate of the local weights. 0.005 by default.
    :param neighborhood_function: (String) The type of neighborhood function to use. "gaussian". by default.
    :param name: (String) The name of this layer for the Tensorflow scope. "som2DGridLayer" by default.
    :param verbose: (Boolean) If true, statistic will be shown in Tensorboard. False by default.
    :param neighborhoodFunction_args: (**) The parameters for the neighborhood function.
    :return: distances_block_batch: (Tensor) the output of the CSNN layer analog to CNNs.
                                    Shape:  [batch_size, num_conv_patches_h, num_conv_patches_w,
                                    num_maps * num_som_neurons_per_map].
    :return: neighborhood_coeff_block_batch: (Tensor) The neighborhood_coeffs of the output.
                                             Shape: [batch_size, num_conv_patches_h, num_conv_patches_w,
                                             num_maps * num_som_neurons_per_map].
    :return: bmus_batch: (Tensor) The BMU cords for the output.
                         Shape: [batch_size*num_conv_patches_h*num_conv_patches_w, num_maps]
    :return: input_patches_batch: (Tensor) The input patches of the layer as batch.
                                  Shape: [batch_size, num_conv_patches_h, num_conv_patches_w, patch_size]
    :return: som_weights: (Tensor) The SOM weights of the layer.
                          Shape: [num_maps, num_som_neurons_per_map, som_neuron_weight_vector_size]
    :return: local_weights: (Tensor) The local weights of the layer.
                            Shape: [num_maps, num_som_neuron, num_som_neurons_prev_layer or patch_size]
    """
    # Check the input parameters
    assert len(input_batch.shape) == 4  # Wanted shape: [batch_size, input_height, input_width, input_depth].
    assert len(som_grid) == 3  # Wanted shape: [som_grid_h, som_grid_w, num_maps].
    assert len(som_kernel) == 2  # Wanted shape: [som_kernel_h, som_kernel_w].
    assert len(strides) == 2  # Wanted shape: stride_h, stride_w].
    assert len(som_kernel) == 2  # Wanted shape: [som_kernel_h, som_kernel_w].
    assert len(train_interval) == 2  # Wanted shape: [som_kernel_h, som_kernel_w].

    # Check if the functions for the given function names exist.
    assert bmu_metrics in bmuMetrics.bmu_metrics_names
    assert neighborhood_function in neighborhoodFunctions.neighborhood_function_names
    assert local_weights_type in local_weights_types
    assert som_learning_rule in learningRules.som_learning_rule_names
    assert local_learning_rule in learningRules.local_learning_rule_names

    # Get needed sizes
    som_grid_height = som_grid[0]
    som_grid_width = som_grid[1]
    num_som_neurons_per_map = som_grid_height * som_grid_width
    num_maps = som_grid[2]
    som_kernel_depth = input_batch.get_shape()[3].value

    # Create the input patches
    # Input Tensor Shape input_batch: [batch_size, input_height, input_width, input_depth]
    # Output Tensor Shape input_patches_batch : [batch_size, num_conv_patches_h, num_conv_patches_w,
    # patch_size*input_depth]
    # Output Tensor Shape input_patches_batch_re : [batch_size=batch_size*num_conv_patches_h*num_conv_patches_w,
    # patch_size=som_kernel[0]*som_kernel[1]*som_kernel_depth]
    # Output Tensor Shape: [1]
    # Output Tensor Shape: [1]
    input_patches_batch, input_patches_batch_re, num_conv_patches_h, num_conv_patches_w, patch_size = \
        createInputPatches(input_batch, som_kernel, som_kernel_depth, strides, padding, name)

    # Create the local weights/mask type
    # Input Tensor Shape input_patches_batch_re : [batch_size, patch_size]
    # Output Tensor Shape input_patches_batch_metric: [batch_size, num_maps or 1, num_som_neuron or 1, patch_size]
    # Output Tensor Shape output_local_update_batch: [batch_size, (kernel_width*kernel_height), num_maps,
    # num_som_neuron, patch_size] or None
    # Output Tensor Shape input_local_update_batch: [batch_size, (kernel_width*kernel_height), 1, 1, patch_size] or None
    # Output Tensor Shape local_weights: [num_maps, num_som_neuron, num_som_neurons_prev_layer or patch_size] or None
    input_patches_batch_metric, output_local_update_batch, input_local_update_batch, local_weights = \
        createLocalWeights(local_weights_type, input_patches_batch_re, num_maps, num_som_neurons_per_map,
                           som_kernel_depth, patch_size, name)

    # Define weights for each neuron per map in the CSNN. These can be seen as defining num_maps soms.
    # These SOMs are used as a filter in a conv net.
    # Tensor Shape: [num_maps, num_som_neurons_per_map, som_neuron_weight_vector_size = patch_size]
    som_weights = tf.get_variable(name+"Weights", [num_maps, num_som_neurons_per_map, patch_size],
                                      initializer=tf.random_uniform_initializer(-1.0, 1.0))

    # Input Tensor Shape: [num_maps, num_som_neuron, num_som_neurons_prev_layer]
    # Output Tensor Shape: [1, num_maps, num_som_neuron, num_som_neurons_prev_layer]
    som_weights_metric = tf.expand_dims(som_weights, axis=0)

    # Compute the metrics between each som neuron and the input for each map.
    # Input Tensor Shape som_net_weights: [num_maps, num_som_neurons_per_map, som_neuron_weight_vector_size]
    # Input Tensor Shape input_patches_re: [batch_size, som_neuron_weight_vector_size]
    # Output Tensor Shape: [batch_size, num_maps] - the bmu for each batch.
    # Output Tensor Shape: [batch_size, num_maps, num_som_neurons] - the "distances" of the som_neurons for each batch.
    bmus_batch, distances_batch = getattr(bmuMetrics, bmu_metrics)(som_weights_metric, input_patches_batch_metric,
                                                                   verbose)
    # Define cords for each neuron per map in the CSNN.
    # Tensor Shape: [num_som_neurons, 2]
    neuron_cords = tf.constant(np.array(list(__neuronCords(som_grid_height, som_grid_width))))

    # Define train and inference functions of the layer. If the layer is currently learning, neighborhood_coeffs and
    # delta weight updates need to be computed. If the layer currently does not learn, the delta weights an
    # neighborhood_coeffs are simply zero.
    def train():
        delta_som_weights, delta_local_weights, neighborhood_coeff_batch = __csnnLayerTrain(neuron_cords,
                        neighborhood_function, bmus_batch, train_interval, distances_batch, som_weights,
                        local_weights,output_local_update_batch, input_local_update_batch, som_learning_rule,
                        local_learning_rule, local_weights_type, local_neighborhood_coeff_type, som_learning_rate,
                        local_learning_rate, input_patches_batch_re, **neighborhoodFunction_args)
        return delta_som_weights, delta_local_weights, neighborhood_coeff_batch

    def infer():
        delta_som_weights, delta_local_weights, neighborhood_coeff_batch = __csnnLayerInfer(som_weights, local_weights,
                                                                                    local_weights_type, distances_batch)
        return delta_som_weights, delta_local_weights, neighborhood_coeff_batch

    # If the global step lies in the training intervall, train the layer, else infer.
    global_step = tf.train.get_or_create_global_step()
    delta_som_weights, delta_local_weights, neighborhood_coeff_batch = tf.cond(
        tf.logical_and(tf.greater_equal(tf.cast(global_step, tf.int32), tf.cast(train_interval[0], tf.int32)),
        tf.less(tf.cast(global_step, tf.int32), tf.cast(train_interval[1], tf.int32))), true_fn=train,
        false_fn=infer)

    # Reshape the distances_batch (and neighborhood_coeff)_batch to a CSNN output blocks equal to the output of a CNN,
    # where the number of feature maps equals the number of som_neurons in this layer times the number of used som maps.
    # The width and height of the block is defined by the used padding, stride and kernel size equal to CNNs.
    # Input Tensor Shape neighborhood_coeff_batch: [batch_size*num_conv_patches_h*num_conv_patches_w, num_maps,
    # num_som_neurons_per_map]
    # Input Tensor Shape distances_batch: [batch_size*num_conv_patches_h*num_conv_patches_w, num_maps,
    # num_som_neurons_per_map]
    # Output Tensor Shape neighborhood_coeff_batch : [batch_size, num_conv_patches_h, num_conv_patches_w,
    # num_maps * num_som_neurons_per_map].
    # Output Tensor Shape distances_batch: [batch_size, num_conv_patches_h, num_conv_patches_w,
    # num_maps * num_som_neurons_per_map].
    distances_block_batch = tf.reshape(distances_batch, [-1, num_conv_patches_h, num_conv_patches_w,
                                                   num_maps*num_som_neurons_per_map])
    neighborhood_coeff_block_batch = tf.reshape(neighborhood_coeff_batch, [-1, num_conv_patches_h, num_conv_patches_w,
                                                          num_maps*num_som_neurons_per_map])

    # Output Tensor Shape delta_som_weights: [num_maps, num_som_neurons_per_map, som_neuron_weight_vector_size]
    # - the delta_weights weights for the training (see optimization).
    # Output Tensor Shape som_net_weights: [num_maps, num_som_neurons_per_map, som_neuron_weight_vector_size]
    tf.add_to_collections("deltaSomWeights", delta_som_weights)
    tf.add_to_collections("oldSomWeights", som_weights)

    if local_weights_type == "betweenNeurons" or local_weights_type == "betweenInputAndNeurons":
        # Output Tensor Shape deltaLocalWeights:[num_maps, num_som_neuron, num_som_neurons_prev_layer or patch_size]
        # - the delta weights of the mask (local) for the training (see optimization).
        # Output Tensor Shape oldLocalWeights: [num_maps, num_som_neuron, num_som_neurons_prev_layer or patch_size]
        tf.add_to_collections("deltaLocalWeights", delta_local_weights)
        tf.add_to_collections("oldLocalWeights", local_weights)

    # Output Tensor Shape: [1]
    neuron_utilization = evaluation.neuronUtilization(neuron_cords, bmus_batch, name)
    tf.add_to_collections("neuronUtilization", neuron_utilization)

    # Output Tensor Shape distances_block_batch : [batch_size, num_conv_patches_h, num_conv_patches_w,
    # num_maps * num_som_neurons_per_map].
    # Output Tensor Shape neighborhood_coeff_block_batch : [batch_size, num_conv_patches_h, num_conv_patches_w,
    # num_maps * num_som_neurons_per_map].
    # Output Tensor Shape bmus_batch: [batch_size*num_conv_patches_h*num_conv_patches_w, num_maps]
    # Output Tensor Shape input_patches_batch: [batch_size, num_conv_patches_h, num_conv_patches_w, patch_size]
    # Output Tensor Shape som_weights: [num_maps, num_som_neurons_per_map, som_neuron_weight_vector_size]
    # Output Tensor Shape local_weights: [num_maps, num_som_neuron, num_som_neurons_prev_layer or patch_size]
    return distances_block_batch, neighborhood_coeff_block_batch, bmus_batch, input_patches_batch, som_weights, \
           local_weights


def chooseNormalization(distances_block_batch, bmu_metrics_value_invert, bmu_metrics_normalization):
    """
    Chooses the normalization given the name of the normalization
    :param distances_block_batch: (Tensor) The output to normalize. Shape : [batch_size, num_conv_patches_h,
                                  num_conv_patches_w, num_maps * num_som_neurons_per_map].
    :param bmu_metrics_value_invert: (Boolean) If true, the maximum value of the input will be the minimum value
                                      and vice versa.
    :param bmu_metrics_normalization: (String) The name of the normalization.
    :return: distances_block_batch: (Tensor) The normalized output. Shape : [batch_size, num_conv_patches_h,
                                    num_conv_patches_w, num_maps * num_som_neurons_per_map].
    """
    if bmu_metrics_value_invert:
        distances_block_batch = tf.add(-distances_block_batch, tf.reduce_max(distances_block_batch, axis=3,
                                                                             keepdims=True))
    if bmu_metrics_normalization == "maxDiv":
        return tf.div(distances_block_batch, tf.reduce_max(tf.abs(distances_block_batch), keepdims=True))

    elif bmu_metrics_normalization == "maxDivChannel":
        return tf.div(distances_block_batch, tf.reduce_max(distances_block_batch, axis=3, keepdims=True))

    elif bmu_metrics_normalization == "maxDivChannelBatchNorm":
        distances = tf.div(distances_block_batch, tf.reduce_max(distances_block_batch, axis=3, keepdims=True))
        return tf.layers.batch_normalization(distances, trainable=False)

    elif bmu_metrics_normalization == "batchNorm":
        return tf.layers.batch_normalization(distances_block_batch, trainable=False)

    elif bmu_metrics_normalization == "maxDivBatchNorm":
        distances_block_batch = tf.div(distances_block_batch, tf.reduce_max(distances_block_batch, keepdims=True))
        return tf.layers.batch_normalization(distances_block_batch, trainable=False)

    elif bmu_metrics_normalization == "batchNormChannel":
        return tf.layers.batch_normalization(distances_block_batch, trainable=False, axis=3)

    return distances_block_batch


def featurePooling(distances_block_batch, som_grid, pool_ksize, stride, pooling_type="MAX"):
    """
    Not tested jet! Defines feature pooling operations.
    :param distances_block_batch: (Tensor) The output to pool. Shape : [batch_size, num_conv_patches_h,
                                  num_conv_patches_w, num_maps * num_som_neurons_per_map].
    :param som_grid: (Array) The SOM-grid used to creat the distances_block_batch
    :param pool_ksize: (Integer) The 1D size of the pooling kernel.
    :param stride: (Integer) The 1D stride used for pooling.
    :param pooling_type: (String) The type of feature pooling to use,
    :return: distances_block_batch: (Tensor) The pooled output. Shape : [batch_size, num_conv_patches_h,
                                  num_conv_patches_w, pooling_result_size].
    """
    assert pooling_type in ["MAX", "AVG"]
    # Input Tensor Shape distances : [batch_size, num_conv_patches_h, num_conv_patches_w, num_maps *
    # num_som_neurons_per_map].
    _, h, w, c = distances_block_batch.get_shape()

    #Output Tensor Shape distances : [batch_size, num_conv_patches_h*num_conv_patches_w* num_maps, grid_h, grid_w].
    distances_block_batch = tf.reshape(distances_block_batch, [-1, h.value* w.value*som_grid[2], som_grid[0],
                                                               som_grid[1]])

    # Output Tensor Shape distances : [batch_size, num_conv_patches_h, num_conv_patches_w, num_maps,
    # pooled_h, pooled_w].
    if pooling_type=="MAX":
        distances_block_batch = tf.layers.max_pooling2d(distances_block_batch, pool_ksize, stride,
                                                        data_format="channels_first")
    elif pooling_type == "AVG":
        distances_block_batch = tf.layers.average_pooling2d(distances_block_batch, pool_ksize, stride,
                                                            data_format="channels_first")

    _, _, pooled_h, pooled_w = distances_block_batch.get_shape()

    # Output Tensor Shape distances : [batch_size, num_conv_patches_h, num_conv_patches_w, num_maps*pooled_h.*pooled_w].
    distances_block_batch = tf.reshape(distances_block_batch, [-1, h.value, w.value,
                                                               som_grid[2]*pooled_h.value*pooled_w.value])

    return distances_block_batch


def choosePoolingLayer(distances_block_batch, layer_config):
    """
    Chooses the pooling layer(s) given the layer configuration.
    :param distances_block_batch: (Tensor) The distance output of the csnn layer.
    :return: layer_config: (Dictionary) The configuration of the layer.
    """
    if layer_config["pooling"]:
        if layer_config["pooling"][2] == "max":
            distances_block_batch = tf.layers.max_pooling2d(distances_block_batch, layer_config["pooling"][0],
                                                            layer_config["pooling"][1], padding="SAME")
        elif layer_config["pooling"][2] == "avg":
            distances_block_batch = tf.layers.average_pooling2d(distances_block_batch, layer_config["pooling"][0],
                                                                layer_config["pooling"][1], padding="SAME")

    if layer_config["featurePooling"] == "depthwise":
        distances_block_batch = tf.nn.max_pool(distances_block_batch, [1, 1, 1, 2], [1, 1, 1, 2], 'valid')
    elif layer_config["featurePooling"]:
        distances_block_batch = featurePooling(distances_block_batch, layer_config["somGrid"],
                                            layer_config["featurePooling"][0], layer_config["featurePooling"][1],
                                            pooling_type=layer_config["featurePooling"][2])
    return distances_block_batch