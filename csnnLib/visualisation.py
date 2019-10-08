import tensorflow as tf

def featureChannelVisualisation(block, indices, name_prefix, rgb_vis=False):
    """
    Visualizes the feature channels/maps of the giben block selected with the given 1D (depth) indices.
    If rgb_vis is true, rgb slices of depth three are used.
    :param block: (Tensor) The output block of a (s)conv layer.
                  Shape: [batch_size, num_conv_patches_h, num_conv_patches_w, num_maps * num_som_neurons_per_map]
    :param indices: (Array) The indices of the maps to visualize. Shape [i_1, i_2, ...]
    :param name_prefix: (String) The name prefix of the Tensorboard log.
    :param rgb_vis: (Boolean) If true, rgb slices of depth three will be visualized. False by default.
    """
    block_depth = block.get_shape()[3].value
    # Input Tensor Shape indices : [n]
    for index in indices:
        if rgb_vis:
            assert (index + 2) < block_depth.value
            index = __expandIndexForRgbVis(index)
        else:
            assert index < block_depth.value
            index = [index]

        # Gather the channel(s) to visualize.
        # Input Tensor Shape block : [batch_size, num_conv_patches_h, num_conv_patches_w,
        # num_maps * num_som_neurons_per_map]
        # Input Tensor Shape index : [1]
        # Output Tensor Shape : [batch_size, num_conv_patches_h, num_conv_patches_w, 1 or 3]
        channel = tf.gather(block, index, axis=3)

        # Show the channel in Tensorboard.
        tf.summary.image(str(name_prefix)+"Channel"+str(index), channel, max_outputs=1)

def csnnSpatialActivationVisualisation(block, indices, grid, name_prefix):
    """
    Visualizes the spatial activation (feature vectors) of every map of the given block selected with the given 2D
    (height, width) indices.
    :param block:  (Tensor) The output block of a sconv layer.
                   [batch_size, num_conv_patches_h, num_conv_patches_w, num_maps * num_som_neurons_per_map]
    :param indices: (Array) The indices of the spatial activations to visualize. Shape [i_1, i_2, ...]
    :param grid: (Array) The grid of the corresponding (s)conv layer. Shape [h, w, maps]
    :param name_prefix: (String) The name prefix of the Tensorboard log.
    """
    # Input Tensor Shape indices : [n,m]
    for index in indices:
        # Gather the height of the index.
        # Input Tensor Shape block : [batch_size, num_conv_patches_h, num_conv_patches_w,
        # num_maps * num_som_neurons_per_map]
        # Input Tensor Shape index : [1]
        # Output Tensor Shape : [batch_size, num_conv_patches_w,  num_maps * num_som_neurons_per_map]
        maps = tf.gather(block, index[0], axis=1)

        # Gather the width of the index. The spatial activation.
        # Input Tensor Shape block : [batch_size, num_conv_patches_w, num_maps * num_som_neurons_per_map]
        # Input Tensor Shape index : [1]
        # Output Tensor Shape : [batch_size, num_maps * num_som_neurons_per_map]
        maps = tf.gather(maps, index[1], axis=1)

        # Reshape the spatial activation to a image for Tensorboard.
        # Input Tensor Shape block : [batch_size, num_maps * num_som_neurons_per_map]
        # Output Tensor Shape : [batch_size, num_maps, som_grid_height, som_grid_width]
        maps = tf.reshape(maps, [-1, grid[2], grid[0], grid[1]])

        # Visualize every map
        for map_index in range(grid[2]):
            # Gather the width of the index. The spatial activation.
            # Input Tensor Shape block : [batch_size, num_maps, som_grid_height, som_grid_width, num_maps]
            # Input Tensor Shape index : [1]
            # Output Tensor Shape : [batch_size, som_grid_height, som_grid_width]
            map = tf.gather(maps, map_index, axis=1)

            # Extend the spatial activation to a image for Tensorboard.
            # Input Tensor Shape block : [batch_size, som_grid_height, som_grid_width]
            # Output Tensor Shape block : [batch_size, som_grid_height, som_grid_width, 1]
            map = tf.expand_dims(map, axis=3)

            # Show the spatial activation in Tensorboard.
            tf.summary.image(str(name_prefix)+"Act"+str(index)+"Map"+str(map_index), map, max_outputs=1)

def csnnWeightGridVisualisation(weights, grid, kernel_height, kernel_width, input_depth=1, input_channel_indices=[0],
                                rgb_vis=False):
    """
    Visualizes the 2D weight grid of every map of the given weights for the input channel(s) selected with the given
    indices.
    The 2D weight grid is a image for every map of a slice (similar to a feature map slice) trough each
    SOM neurons weights kernel. These slices are then brought together into an image corresponding to the SOM neurons
    2D grid position.

    These visualization is very useful for debugging the CSNN, because it shows if the hyperparameters
    (e.g. neighborhood coefficient and grid size) match and ever neuron is learned.

    :param weights: (Tensor) The weights to visualize.
                    Shape  [num_som_neurons_per_map, som_neuron_weight_vector_size]
    :param grid: (Array) The grid of the corresponding sconv layer. Shape [h, w, maps]
    :param kernel_height: (Integer) The height of the som neuron kernel.
    :param kernel_width: (Integer) The width of the som neuron kernel.
    :param input_depth: (Integer) The depth of the som neuron kernel/input. 1 by default.
    :param input_channel_indices: (Integer) The indices to select the slices trough the som neuron kernel
                                  (similar to a feature map slice) [0] by default.
    :param rgb_vis: (Boolean) If true, rgb slices of depth three will be visualized. False by default.
    """
    som_grid_height = grid[0]
    som_grid_width = grid[1]

    # Reshaping the weights to compute the weight grid.
    # Input Tensor Shape: [num_som_neurons_per_map, som_neuron_weight_vector_size] - The weight from one map
    # Output Tensor Shape: [som_grid_height, som_grid_width, kernel_height, kernel_width, input_depth]
    weight_patches = tf.reshape(weights, [som_grid_height, som_grid_width, kernel_height, kernel_width, input_depth])

    # Transpose the weights to compute the weight grid.
    # Input Tensor Shape: [som_grid_height, som_grid_width, input_height, kernel_width, input_depth]
    # - The weight from one map.
    # Output Tensor Shape: [som_grid_height, kernel_height, som_grid_width, kernel_width, input_depth]
    weight_patches = tf.transpose(weight_patches, [0, 2, 1, 3, 4])

    # Reshape the weights for the final weight grid.
    # Input Tensor Shape: [som_grid_height, input_height, som_grid_width, kernel_height, input_depth]
    # - The weight from one map.
    # Output Tensor Shape: [1, som_grid_height * kernel_height, som_grid_width * kernel_width, input_depth]
    weight_patches = tf.reshape(weight_patches, [1, som_grid_height * kernel_height, som_grid_width * kernel_width,
                                                 input_depth])

    # Vis the weights for the given depths.
    for index in input_channel_indices:

        if rgb_vis:
            assert (index+2) < input_depth
            index = __expandIndexForRgbVis(index)
        else:
            assert index < input_depth
            index = [index]

        # Gather the weight for the given depth.
        # Input Tensor Shape: [1, som_grid_height * kernel_height, som_grid_width * kernel_width, input_depth]
        # Output Tensor Shape: [1, som_grid_height * kernel_height, som_grid_width * kernel_width, 1 or 3]
        weight_channel = tf.gather(weight_patches, index, axis=3)

        # Vis the weight in Tensorboard.
        tf.summary.image("WeightGridForChannel" + str(index), weight_channel, max_outputs=1)


def csnnWeightImageVisualisation(weights, bmus, grid, som_kernel, num_conv_patches_h, num_conv_patches_w,
                                 input_channel_indices=[0], map_index=0, rgb_vis=False, neighborhood_coeff_batch=None,
                                 distances_batch=None):
    """
    Constructs a image by selecting the BMUs weight for each patch and putting the patches together to one image.
    Strides and padding are not taken into account, therefore the image may have a different size then the input image.
    The weights depth dimension may be bigger then 3, therefore slice (similar to a feature map slice) will be
    visualized. The slices can be selected trough given map and depth indices.
    :param weights: (Tensor) The weights of the som neurons.
                    Shape  [num_som_neurons_per_map, som_neuron_weight_vector_size]
    :param bmus: (Array) The BMU indices for the corresponding patches.
    :param grid: (Array) The grid of the corresponding sconv layer. Shape [h, w, maps]
    :param som_kernel: (Array) The kernel of the corresponding sconv layer. Shape [h, w]
    :param num_conv_patches_h: (Integer) The number of patches in the height dimension.
    :param num_conv_patches_w: (Integer) The number of patches in the width dimension.
    :param input_channel_indices: (Integer) The indices to select the slices trough the BMU kernel
                                  (similar to a feature map slice) [0] by default.
    :param map_index: (Integer) The indices to select the map. If -1 the best BMU of all maps is taken. Only supports
                       argmax distance metrics!
    :param rgb_vis: (Boolean) If true, rgb slices of depth three will be visualized. False by default.
    :param distances_batch: (Tensor) The distances of the SOM neurons to the CSNN layer patches.
                            Shape: [batch_size, num_maps, num_som_neurons]. None by default.
    :param neighborhood_coeff_batch: (Tensor) The neighborhood_coeffs of the SOM neurons.
                                     Shape: [batch_size, num_maps, num_som_neurons]. None by default.
    """
    if map_index is -1:
        # Input Tensor Shape: [batch_size, num_maps, num_som_neurons]
        # Output Tensor Shape: [batch_size]
        max_map_index_batch = tf.math.argmax(tf.reduce_max(distances_batch, axis=2), axis=1)

        # Input Tensor Shape: [batch_size]
        # Output Tensor Shape: [batch_size, num_maps]
        maps = neighborhood_coeff_batch.get_shape()[1].value
        mask_batch = tf.one_hot(max_map_index_batch, maps)

        # Input Tensor Shape neighborhood_coeff_batch: [batch_size, num_maps]
        # Input Tensor Shape mask_batch: [batch_size, num_maps]
        # Output Tensor Shape: [batch_size, num_maps]
        bmus = tf.multiply(bmus, mask_batch)

    # Gather the weights for given bmus.
    # Input Tensor Shape weigths_re: [num_maps, num_som_neurons_per_map, som_neuron_weight_vector_size]
    # Input Tensor Shape bmus: [batch_size*num_conv_patches_h*num_conv_patches_w, num_maps]
    # Output Tensor Shape: [batch_size*num_conv_patches_h*num_conv_patches_w, num_maps, som_neuron_weight_vector_size]
    weigths_for_bmus = tf.gather(weights, tf.reshape(bmus, [-1]), axis=1)
    weigths_for_bmus = tf.transpose(weigths_for_bmus, [1, 0, 2])

    # Reshape the weight.
    # Input Tensor Shape weigths_for_bmus: [batch_size*num_conv_patches_h*num_conv_patches_w, num_maps,
    # som_neuron_weight_vector_size]
    # Output Tensor Shape: [batch_size, num_conv_patches_h, num_conv_patches_w, num_maps, som_kernel[0],
    # som_kernel[1], kernel_depth]
    _, _, som_vec_size = weigths_for_bmus.get_shape()
    weigths_for_bmus = tf.reshape(weigths_for_bmus, [-1, num_conv_patches_h, num_conv_patches_w, grid[2], som_kernel[0],
                                                     som_kernel[1], som_vec_size.value//(som_kernel[0]*som_kernel[1])])

    # Transpose the weight.
    # Input Tensor Shape weigths_re: [batch_size, num_conv_patches_h, num_conv_patches_w, num_maps, som_kernel[0],
    # som_kernel[1], input_depth]
    # Output Tensor Shape: [batch_size, num_maps, num_conv_patches_h, som_kernel[0], num_conv_patches_w , som_kernel[1],
    # kernel_depth]
    weigths_for_bmus = tf.transpose(weigths_for_bmus, [0, 3, 1, 4, 2, 5, 6])

    # Reshape the weight for weight image per batch and map
    # Input Tensor Shape weigths_re: [batch_size, num_maps, num_conv_patches_h, som_kernel[0], num_conv_patches_w,
    # som_kernel[1], kernel_depth]
    # Output Tensor Shape: [batch_size, num_maps, num_conv_patches_h*som_kernel[0], num_conv_patches_w*som_kernel[1],
    # kernel_depth]
    weigths_block = tf.reshape(weigths_for_bmus, [-1, grid[2], num_conv_patches_h*som_kernel[0],
                                                  num_conv_patches_w*som_kernel[1],
                                                  som_vec_size.value//(som_kernel[0]*som_kernel[1])])

    if map_index is -1:
        weigths_map_block = tf.reduce_sum(weigths_block, axis=1)
    else:
        # Gather the map to vis the weights from.
        # Input Tensor Shape weigths_re: [batch_size, num_maps, num_conv_patches_h*som_kernel[0],
        # num_conv_patches_w*som_kernel[1], input_depth]
        # Output Tensor Shape: [batch_size, num_conv_patches_h*som_kernel[0], num_conv_patches_w*som_kernel[1],
        # kernel_depth]
        weigths_map_block = tf.gather(weigths_block, map_index, axis=1)

    block_depth = weigths_map_block.get_shape()[3].value
    for index in input_channel_indices:
        if rgb_vis:
            assert (index + 2) < block_depth
            index = __expandIndexForRgbVis(index)
        else:
            assert index < block_depth
            index = [index]

        # Gather the depth to visualize the weight.
        # Input Tensor Shape weigths_re:  [batch_size, num_conv_patches_h*som_kernel[0],
        # num_conv_patches_w*som_kernel[1], kernel_depth]
        # Output Tensor Shape: [batch_size, num_conv_patches_h*som_kernel[0], num_conv_patches_w*som_kernel[1], 1 or 3]
        weight_img_channel = tf.gather(weigths_map_block, index, axis=3)

        # Vis the weight in tensorboard.
        tf.summary.image("WeightImgForMap"+str(map_index)+"Channel" + str(index), weight_img_channel, max_outputs=10)

def csnnNeuronLocalWeightVisualisation(local_weights, indices, prev_grid, type, name_prefix="LocalWeights"):
    """
    Visualizes the local weights (mask) of one neuron given by the indices.
    The weights depth dimension may be bigger then 3, therefore slice (similar to a feature map slice) will be taken.
    :param local_weights: (Tensor) The weights of the som neurons.
                           Shape  [num_som_neurons_per_map, som_neuron_weight_vector_size]
    :param indices: (Array) The indices of the som neuron to show the local masks for. Shape: [h, w, map]
    :param prev_grid: (Array) The grid of the previous sconv layer or input. Shape [h, w, d/maps]
    :param type: (String) The type of local weights (mask) used in the layer.
    :param name_prefix: (String) The name prefix of the Tensorboard log.
    """
    # Input Tensor Shape indizes : [m, n]
    # Gather the height of the index.
    # Input Tensor Shape block : [num_maps, num_som_neuron, num_som_neurons_prev_layer or som_neuron_weight_vector_size]
    # Input Tensor Shape index : [1]
    # Output Tensor Shape : [num_som_neuron, num_som_neurons_prev_layer or som_neuron_weight_vector_size]
    map = tf.gather(local_weights, indices[0], axis=0)

    # Gather the width of the index. The spatial activation.
    # Input Tensor Shape map : [num_som_neuron, num_som_neurons_prev_layer or som_neuron_weight_vector_size]
    # Input Tensor Shape index : [1]
    # Output Tensor Shape : [num_som_neurons_prev_layer or som_neuron_weight_vector_size]
    neuron = tf.gather(map, indices[1], axis=0)

    # Reshape the spatial activation to a image for tensorboard.
    # Input Tensor Shape neuron : [num_som_neurons_prev_layer or som_neuron_weight_vector_size]
    # Output Tensor Shape : [prev_num_maps, prev_som_grid_height, prev_som_grid_width, kernel_depth or 1]
    neuron = tf.reshape(neuron, [prev_grid[2], prev_grid[0], prev_grid[1], -1])
    _, _, _, kernel_depth = neuron.get_shape()

    for map_index in range(prev_grid[2]):
        # Gather the width of the index. The neuron local weights.
        # Input Tensor Shape block : [prev_num_maps, prev_som_grid_height, prev_som_grid_width, kernel_depth or 1]
        # Input Tensor Shape index : [1]
        # Output Tensor Shape : [prev_som_grid_height, prev_som_grid_width, kernel_depth or 1]
        map = tf.gather(neuron, map_index, axis=0)
        if type == "betweenNeurons":
            # Show the spatial activation in Tensorboard.
            tf.summary.image(str(name_prefix) + "Map" +str(indices[0])+"Neuron" + str(indices[1]) + "PrevMap" +
                             str(map_index), tf.expand_dims(map, axis=0), max_outputs=1)
        else:
            for map_part_index in range(kernel_depth.value):
                # Gather the width of the index. The neuron local weights.
                # Input Tensor Shape block : [prev_num_maps, prev_som_grid_height, prev_som_grid_width,
                # kernel_depth or 1]
                # Input Tensor Shape index : [1]
                # Output Tensor Shape : [prev_som_grid_height, prev_som_grid_width, kernel_depth or 1]
                map_part = tf.gather(map, map_part_index, axis=2)

                # Extend the neuron local weights to a image for Tensorboard.
                # Input Tensor Shape block : [1, som_grid_height, som_grid_width]
                # Output Tensor Shape block : [1, som_grid_height, som_grid_width, 1]
                map_part = tf.expand_dims(tf.expand_dims(map_part, axis=0), axis=3)

                # Show the spatial activation in Tensorboard.
                tf.summary.image(str(name_prefix) +"Map"+str(indices[0])+"Neuron" +str(indices[1])+"PrevMap"
                                 + str(map_index), map_part, max_outputs=1)

def __expandIndexForRgbVis(index):
    """
    Expands the given 1D index into a 3D index fpr rgb visualization.
    :param index: (Integer) The index to expand.
    :return: (Array) The expanded 3D index.
    """
    return [index, index+1, index+2]






