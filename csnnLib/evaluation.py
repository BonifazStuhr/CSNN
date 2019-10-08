import tensorflow as tf

"""
Defines functions for the evaluation of the CSNN.
"""

def neuronUtilization(neuron_cords, bmu_batch, name=""):
    """
    Calculates the neuron utilization of the batch. One can use this function to get a feeling if som neurons are dead.
    :param neuron_cords: (Tensor) The cords for each neuron. Shape: [num_som_neurons, 2]
    :param bmu_batch: (Tensor) The batch containing the BMUs. Shape: [batch_size, num_maps]
    :param name: (String) The name of this operation. "" by default.
    :return: neuron_utilization: (Tensor) The neuron utilization of the batch. Shape: [1]
    """
    with tf.variable_scope('neuronUtilization' + str(name)):
        # Compute the cords for each bmu.
        # Input Tensor Shape bmu_batch: [batch_size, num_maps]
        # Input Tensor Shape neuron_cords: [num_som_neurons, 2]
        # Output Tensor Shape: [batch_size, num_maps, 2] - the cords for each bmu.
        bmu_cords_per_map_batch = tf.gather(neuron_cords, bmu_batch)

        # Input Tensor Shape: [batch_size, num_maps, 2]
        # Output Tensor Shape: [batch_size, num_maps, 1, 2]
        bmu_cords_per_map_batch = tf.expand_dims(bmu_cords_per_map_batch, axis=2)

        # Input Tensor Shape: [num_som_neurons, 2]
        # Output Tensor Shape: [1, 1, num_som_neurons 2]
        neuron_cords = tf.expand_dims(tf.expand_dims(neuron_cords, axis=0), axis=0)

        # Compare the coords from the bmu_batch with the cords from the map and set similar cords to true.
        # Input Tensor Shape: [batch_size, num_maps, 1, 2]
        # Input Tensor Shape: [1, 1, num_som_neurons 2]
        # Output Tensor Shape: [batch_size, num_maps, num_som_neurons, 2]
        comp = tf.equal(bmu_cords_per_map_batch, neuron_cords)

        # Reduce the min value of the 2D cord to get True values everywhere the 2D cord matched with the BMU
        # Then reduce the whole batch to get a map with True on every position a BMU was found.
        # Input Tensor Shape: [batch_size, num_maps, num_som_neurons, 2]
        # Output Tensor Shape: [num_maps, num_som_neurons]
        unique = tf.reduce_max(tf.reduce_min(tf.to_int32(comp), axis=3), axis=0)

        # Count the values set to true.
        # Input Tensor Shape: [num_maps, num_som_neurons]
        # Output Tensor Shape: [1]
        non_zeros = tf.math.count_nonzero(unique)

        # Calculate the number of neurons and div the number of used neurons trough all neurons in the layer.
        # Input Tensor Shape: [1]
        # Output Tensor Shape: [1]
        num_maps, nr_neurons = unique.get_shape()
        neuron_utilization = tf.truediv(tf.to_float(non_zeros), tf.to_float(num_maps.value*nr_neurons.value))

        # Output Tensor Shape: [1]
        return neuron_utilization
