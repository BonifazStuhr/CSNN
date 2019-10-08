import tensorflow as tf

from csnnLib import decayFunctions

"""
Defines neighborhood functions for the SOM.
"""

neighborhood_function_names = ["gaussian", "gaussian_allBmus"]


def gaussian(neuron_cords, bmu_batch, training_steps=None, std_coeff=1.0, decrease_neighborhood_range=False):
    """
    Gaussian neighborhood as described in our paper.

    :param neuron_cords: (Tensor) The cords for each neuron. Shape: [num_som_neurons, 2]
    :param bmu_batch: (Tensor) The batch containing the BMUs. Shape: [batch_size, num_maps]
    :param training_steps: (Float) The maximum steps to train. None by default.
    :param std_coeff: (Float) The coeff for the Gaussian neighborhood. 1.0 by default.
    :param decrease_neighborhood_range: (Boolean) If true the range of the neighborhood will be decreased relativ to the
          training steps. False by default.
    :return:
    """
    with tf.variable_scope("gaussian"):
        # Compute the cords for each bmu.
        # Input Tensor Shape bmus: [batch_size, num_maps]
        # Input Tensor Shape neuron_cords: [num_som_neurons, 2]
        # Output Tensor Shape: [batch_size, num_maps, 2] - the cords for each bmu.
        bmu_cords_per_map_batch = tf.gather(neuron_cords, bmu_batch)

        # Input Tensor Shape neuron_cords: [batch_size, num_maps, 2]
        # Output Tensor Shape neuron_cords: [batch_size, num_maps, 1, 2]
        bmu_cords_per_map_batch = tf.expand_dims(bmu_cords_per_map_batch, axis=2)

        # Input Tensor Shape neuron_cords: [num_som_neurons, 2]
        # Output Tensor Shape neuron_cords: [1, 1, num_som_neurons, 2]
        neuron_cords = tf.expand_dims(tf.expand_dims(neuron_cords, axis=0), axis=1)

        # Input Tensor Shape neuron_cords: [batch_size, num_maps, 1, 2]
        # Input Tensor Shape neuron_cords: [1, 1, num_som_neurons, 2]
        # Output Tensor Shape neuron_cords: [batch_size, num_maps, num_som_neurons, 2]
        bmu_distances_batch = tf.to_float(tf.subtract(neuron_cords, bmu_cords_per_map_batch))

        # Input Tensor Shape neuron_cords: [batch_size, num_maps, num_som_neurons, 2]
        # Output Tensor Shape neuron_cords: [batch_size, num_maps, num_som_neurons]
        euclidean_distances_batch = tf.norm(bmu_distances_batch, axis=3)
        neg_squard_euclidean_distances_batch = tf.negative(tf.square(euclidean_distances_batch))

        # Decrease neighborhood to maximum 1.0 if true.
        if decrease_neighborhood_range:
            global_step = tf.to_float(tf.train.get_or_create_global_step())
            std_coeff = decayFunctions.decreaseCoeff(std_coeff, 1.0, global_step, training_steps)

        # Compute the neighborhood_coeff for each neuron.
        # Input Tensor Shape neg_squard_euclidean_distances_batch: [batch_size, num_maps, num_som_neurons]
        # Input Tensor Shape std_coeff: [1]
        # After square and multiply std_coeff: Tensor Shape: [1]
        # Output Tensor Shape: [batch_size, num_maps, num_som_neurons].
        neighborhood_coeff_batch = tf.exp(tf.div(neg_squard_euclidean_distances_batch,
                                                 tf.multiply(2.0, tf.square(std_coeff))))

        # Output Tensor Shape: [batch_size, num_maps, num_som_neurons] - the neighborhood_coeff for each neuron.
        return neighborhood_coeff_batch
