import tensorflow as tf

"""
Defines distance functions (e.g. convolution) used by the sconv layer to calculate the distance from each som neuron to 
the input. These functions also return the BMU per som map.
"""

bmu_metrics_names = ["manhattanDistance", "euclideanDistance", "pearsonCorrelation", "convolutionDistance"]

def manhattanDistance(input_batch, weights, verbose):
    """
    Calculates the Manhattan distance between the given weights and the input.
    :param input_batch: (Tensor) The input_batch to calculate the distance with.
                        Shape: [batch_size, 1 or num_maps, 1 or num_som_neurons, weight_vector_size]
    :param weights: (Tensor) The weights to calculate the distance with.
                    Shape: [1, num_maps, num_som_neurons, som_neuron_weight_vector_size]
    :param verbose: (Boolean) If true, the sum of distances will be added to the collection sum_distance_measure
    :return: bmus_per_map_batch: (Tensor) The BMU for each map form each input sample in the batch
                                 Shape: [batch_size, num_maps]
    :return: manhattan_distances_per_map_batch: (Tensor) The Manhattan distances for each map form each input sample
                                                in the batch to the given weights.
                                                Shape: [batch_size, num_maps, num_som_neurons]
    """
    with tf.variable_scope('manhattanDistance'):
        # Compute the distance_vectors between each weight (or SOM neuron) and the input_batch per map.
        # Input Tensor Shape weights: [1, num_maps, num_som_neurons, weight_vector_size]
        # Input Tensor Shape input_batch:  [batch_size, 1 or num_maps, 1 or num_som_neurons, weight_vector_size]
        # Output Tensor Shape: [batch_size, num_maps, num_som_neurons, weight_vector_size]
        distance_vectors_per_map_batch = tf.subtract(weights, input_batch)

        # Compute the manhattan distance between each weight (or SOM neuron) and the input_batch per map.
        # Input Tensor Shape: [batch_size, num_maps, num_som_neurons, weight_vector_size]
        # Output Tensor Shape: [batch_size, num_maps, num_som_neurons]
        manhattan_distances_per_map_batch = tf.norm(distance_vectors_per_map_batch, axis=3, ord=1)

        # Compute the BMUs with the minimum distance to the input_batches for each map.
        # Input Tensor Shape: [batch_size, num_maps, num_som_neurons]
        # Output Tensor Shape: [batch_size, num_maps]
        bmus_per_map_batch = tf.argmin(manhattan_distances_per_map_batch, axis=2)

        # Input Tensor Shape: [batch_size, num_maps, num_som_neurons]
        # Output Tensor Shape: [1]
        if verbose:
            min_distances_per_map = tf.reduce_min(manhattan_distances_per_map_batch, axis=2)
            min_distance_sum = tf.reduce_sum(min_distances_per_map)
            tf.add_to_collections("sum_distance_measure", min_distance_sum)

        # Output Tensor Shape: [batch_size, num_maps]
        # Output Tensor Shape: [batch_size, num_maps, num_som_neurons]
        return bmus_per_map_batch, manhattan_distances_per_map_batch


def euclideanDistance(input_batch, weights, verbose):
    """
    Calculates the Euclidean distance between the given weights and the input.
    :param input_batch: (Tensor) The input_batch to calculate the distance with.
                        Shape: [batch_size, 1 or num_maps, 1 or num_som_neurons, weight_vector_size]
    :param weights: (Tensor) The weights to calculate the distance with.
                    Shape: [1, num_maps, num_som_neurons, som_neuron_weight_vector_size]
    :param verbose: (Boolean) If true, the sum of distances will be added to the collection sum_distance_measure
    :return: bmus_per_map_batch: (Tensor) The BMU for each map form each input sample in the batch
                                 Shape: [batch_size, num_maps]
    :return: euclidean_distances_per_map_batch: (Tensor) The Euclidean distance for each map form each input sample
                                                in the batch to the given weights.
                                                Shape: [batch_size, num_maps, num_som_neurons]
    """
    with tf.variable_scope('euclideanDistance'):
        # Compute the distance_vectors between each weight (or SOM neuron) and the input_batch per map.
        # Input Tensor Shape weights: [1, num_maps, num_som_neurons, weight_vector_size]
        # Input Tensor Shape input_batch: [batch_size, 1 or num_maps, 1 or num_som_neurons, weight_vector_size]
        # Output Tensor Shape: [batch_size, num_maps, num_som_neurons, weight_vector_size]
        distance_vectors_per_map_batch = tf.subtract(weights, input_batch)

        # Compute the Euclidean distance between each weight (or SOM neuron) and the input_batch per map.
        # Input Tensor Shape: [batch_size, num_maps, num_som_neurons, weight_vector_size]
        # Output Tensor Shape: [batch_size, num_maps, num_som_neurons]
        euclidean_distances_per_map_batch = tf.norm(distance_vectors_per_map_batch, axis=3)

        # Compute the BMUs with the minimum distance to the input_batches for each map.
        # Input Tensor Shape: [batch_size, num_maps, num_som_neurons]
        # Output Tensor Shape: [batch_size, num_maps]
        bmus_per_map_batch = tf.argmin(euclidean_distances_per_map_batch, axis=2)

        # Input Tensor Shape: [batch_size, num_maps, num_som_neurons]
        # Output Tensor Shape: [1]
        if verbose:
            min_distances_per_map = tf.reduce_min(euclidean_distances_per_map_batch, axis=2)
            min_distance_sum = tf.reduce_sum(min_distances_per_map)
            tf.add_to_collections("sum_distance_measure", min_distance_sum)

        # Output Tensor Shape: [batch_size, num_maps]
        # Output Tensor Shape: [batch_size, num_maps, num_som_neurons]
        return bmus_per_map_batch, euclidean_distances_per_map_batch


def pearsonCorrelation(input_batch, weights, verbose):
    """
    Calculates the Pearson Correlation between the given weights and the input.
    :param input_batch: (Tensor) The input_batch to calculate the distance with.
                        Shape: [batch_size, 1 or num_maps, 1 or num_som_neurons, weight_vector_size]
    :param weights: (Tensor) The weights to calculate the distance with.
                    Shape: [1, num_maps, num_som_neurons, som_neuron_weight_vector_size]
    :param verbose: (Boolean) If true, the sum of distances will be added to the collection sum_distance_measure
    :return: bmus_per_map_batch: (Tensor) The BMU for each map form each input sample in the batch
                                 Shape: [batch_size, num_maps]
    :return: pearson_correlation_per_map_batch: (Tensor) The Pearson Correlation for each map form each input sample
                                                in the batch to the given weights.
                                                Shape: [batch_size, num_maps, num_som_neurons]
    """
    with tf.variable_scope('pearsonCorrelation'):

        # There is a formal identity between the correlation coefficient, and the cosine of the angle between two
        # random vectors when x and y have zero mean
        #person_r_per_map = tf.div_no_nan(tf.reduce_sum(tf.multiply(input, som_net_weights), axis=3),
        #                                 tf.multiply(tf.norm(input), tf.norm(som_net_weights)))

        # Compute the covariance between x and y.
        # Input Tensor Shape x: [1 or batch_size, 1 or num_maps, 1 or num_som_neurons, weight_vector_size]
        # Input Tensor Shape y: [1 or batch_size, 1 or num_maps, 1 or num_som_neurons, weight_vector_size]
        # Output Tensor Shape: [1 or batch_size, 1 or num_maps, 1 or num_som_neurons, weight_vector_size]
        def covariance(x, y):
            x_mean = tf.reduce_mean(x, axis=3, keepdims=True)
            y_mean = tf.reduce_mean(y, axis=3, keepdims=True)
            return tf.reduce_sum(tf.multiply(tf.subtract(x, x_mean), tf.subtract(y, y_mean)), axis=3)

        # Input Tensor Shape weights: [1, num_maps, num_som_neurons, weight_vector_size]
        # Input Tensor Shape input_batch: [batch_size, 1 or num_maps, 1 or num_som_neurons, weight_vector_size]
        # Output Tensor Shape cov: [batch_size, num_maps, num_som_neurons, 1]
        # Output Tensor Shape var_input: [batch_size, 1 or num_maps, 1 or num_som_neurons, 1]
        # Output Tensor Shape var_weights: [1, num_maps, num_som_neurons, 1]
        cov = covariance(input_batch, weights)
        var_input = covariance(input_batch, input_batch)
        var_weights = covariance(weights, weights)

        # Input Tensor Shape cov: [batch_size, num_maps, num_som_neurons, 1]
        # Input Tensor Shape var_input: [batch_size, 1 or num_maps, 1 or num_som_neurons, 1]
        # Input Tensor Shape var_weights: [1, num_maps, num_som_neurons, 1]
        # Tensor Shape after multiply: [batch_size, num_maps, num_som_neurons]
        # Output Tensor Shape: [batch_size, num_maps, num_som_neurons]
        person_r_per_map_batch = tf.div_no_nan(cov, tf.multiply(tf.sqrt(var_input), tf.sqrt(var_weights)))

        # Compute the BMUs with the maximum correlation to the input_batches for each map.
        # Input Tensor Shape: [batch_size, num_maps, num_som_neurons]
        # Output Tensor Shape: [batch_size, num_maps]
        bmus_per_map_batch = tf.argmax(person_r_per_map_batch, axis=2)

        # Input Tensor Shape: [batch_size, num_maps, num_som_neurons]
        # Output Tensor Shape: [1]
        if verbose:
            max_distances_per_map = tf.reduce_max(person_r_per_map_batch, axis=2)
            max_distance_sum = tf.reduce_sum(max_distances_per_map)
            tf.add_to_collections("sum_distance_measure", max_distance_sum)

        # Output Tensor Shape: [batch_size, num_maps]
        # Output Tensor Shape: [batch_size, num_maps, num_som_neurons]
        return bmus_per_map_batch, person_r_per_map_batch


def convolutionDistance(input_batch, weights, verbose):
    """
    Calculates the convolution distance between the given weights and the input.
    :param input_batch: (Tensor) The input_batch to calculate the distance with.
                        Shape: [batch_size, 1 or num_maps, 1 or num_som_neurons, weight_vector_size]
    :param weights: (Tensor) The weights to calculate the distance with.
                    Shape: [1, num_maps, num_som_neurons, som_neuron_weight_vector_size]
    :param verbose: (Boolean) If true, the sum of distances will be added to the collection sum_distance_measure
    :return: bmus_per_map_batch: (Tensor) The BMU for each map form each input sample in the batch
                                 Shape: [batch_size, num_maps]
    :return: convolution_distance_per_map_batch: (Tensor) The convolution Correlation for each map form each input
                                                sample in the batch to the given weights.
                                                Shape: [batch_size, num_maps, num_som_neurons]
    """
    with tf.variable_scope('convolutionDistance'):
        # Compute the convolution distance between each weight (or SOM neuron) and the input_batch per map.
        # Input Tensor Shape weights:  [1, num_maps, num_som_neurons, weight_vector_size]
        # Input Tensor Shape input_batch: [batch_size, 1 or num_maps, 1 or num_som_neurons, weight_vector_size]
        # Tensor Shape after multiply: [batch_size, num_maps, num_som_neurons, weight_vector_size]
        # Output Tensor Shape: [batch_size, num_maps, num_som_neurons]
        convolution_distance_per_map_batch = tf.reduce_sum(tf.multiply(weights, input_batch), axis=3)

        # Compute the BMUs with the maximum distance to the input_batches for each map.
        # Input Tensor Shape: [batch_size, num_maps, num_som_neurons]
        # Output Tensor Shape: [batch_size, num_maps]
        bmus_per_map_batch = tf.argmax(convolution_distance_per_map_batch, axis=2)

        # Input Tensor Shape: [batch_size, num_maps, num_som_neurons]
        # Output Tensor Shape: [1]
        if verbose:
            max_distances_per_map = tf.reduce_max(convolution_distance_per_map_batch, axis=2)
            max_distance_sum = tf.reduce_sum(max_distances_per_map)
            tf.add_to_collections("sum_distance_measure", max_distance_sum)

        # Output Tensor Shape: [batch_size, num_maps]
        # Output Tensor Shape: [batch_size, num_maps, num_som_neurons]
        return bmus_per_map_batch, convolution_distance_per_map_batch
