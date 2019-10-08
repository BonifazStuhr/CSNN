import copy
import tensorflow as tf

from csnnLib import layers
from csnnLib.optimization import optimize_csnn
from csnnLib import visualisation

class Csnn():
    """
    Configurable version of the CSNN described in our paper.

    Hyperparameters to configure:
        number of layers
        layer_wise_bottom_up_training (if true layers will be learn bottom up)
        per layer:
            kernel size
            som_grid (number of neurons and maps e.g. [10,10,3] 3 2D maps with 100 neurons)
            strides
            padding
            train_interval (e.g. [0,10000] steps)
            bmu_metrics (e.g convolution)
            som_learning_rule
            som_learning_rate
            local_weights_type (mask type type e.g. input or neuron mask)
            local_learning_rule (mask_learning_rule e.g. our learning rule)
            local_learning_rate
            local_neighborhood_coeff_type
            neighborhood_function (gaussian),
            std_coeff (for the neighborhood_function)
            decrease_neighborhood_range(True of False)

    :Attributes:
        __model_config:    (Dictionary) The configuration of the CSNN.
        __layers:          (Dictionary) The configuration of each layer.
    """
    def __init__(self, model_config):
        """
        Constructor, initialize member variables.
        :param model_config: (Dictionary) The configuration of the CSNN.
        """
        self.__model_config = model_config
        self.__layers = model_config["layers"]

        self.__layer_train_intervals = []

        # If layerWiseBottomUpTraining is configured make sure that the training intervals of each layer reach from
        # start to the end of the training.
        for i in range(len(self.__layers)):
            if not self.__model_config["layerWiseBottomUpTraining"]:
                self.__layers[i]["trainInterval"][0] = 0
                self.__layers[i]["trainInterval"][1] = model_config["trainingSteps"]
            self.__layer_train_intervals.append(self.__layers[i]["trainInterval"])

        self.__rgb_vis = True

    def __csnnFn(self, input_batch, is_training):
        """
        Constructs the core CSNN Tensorflow graph.
        :param input_batch: (Tensor) The input batch tensor.
        :param is_training: (Boolean) If true, the training graph is constructed, if false the validation graph.
        :return: distances_per_layer_ops: (Tensor) The distances (representation) for each layer.
        :return: image_patches_op: (Tensor) The image patches of the first layer.
        """
        skip_out = None

        # CSNN
        # Input Tensor Shape: [batch_size, h, w, d] - The image batch.
        # Output Tensor Shape distances_per_layer: [num_layers, batch_size, layeri_h, layeri_w, layeri_num_som_neurons]
        # - The distances of each layer (the sconv feature maps)
        # Output Tensor Shape input_patches_batch_out: [batch_size, num_conv_patches_h, num_conv_patches_w, patch_size]
        # - The patches of the input of the first layer.
        with tf.variable_scope('Csnn'):
            # for vis
            distances_per_layer = []
            input_patches_batch_out=None

            # Define each layer
            for i in range(len(self.__layers)):
                # Only decrease the neighborhood range of the som filters if mode is training.
                decrease_neighborhood_range = self.__layers[i]["decreaseNeighborhoodRange"]
                if not is_training:
                    decrease_neighborhood_range = False

                # Layer i
                # if i=0: Input Tensor Shape: [batch_size, h, w, d] - The image batch.
                # else: Input Tensor Shape: [batch_size, layeri-1_h, layeri-1_w, layeri-1_num_som_neurons] -
                # distances_block_batch of the last layer
                # - The output of layer i-1.
                # Output Tensor Shape: [batch_size, layeri_h, layeri_w, layeri_num_filter_som_neurons]
                # - The distances of the layer (the sconv feature map)
                with tf.variable_scope('Layer'+str(i)):

                    # Output Tensor Shape distances_block_batch : [batch_size, num_conv_patches_h, num_conv_patches_w,
                    # num_maps * num_som_neurons_per_map].
                    # Output Tensor Shape neighborhood_coeff_block_batch : [batch_size, num_conv_patches_h,
                    # num_conv_patches_w, num_maps * num_som_neurons_per_map].
                    # Output Tensor Shape bmus_batch: [batch_size*num_conv_patches_h*num_conv_patches_w, num_maps]
                    # Output Tensor Shape input_patches_batch: [batch_size, num_conv_patches_h, num_conv_patches_w,
                    # patch_size]
                    # Output Tensor Shape som_weights: [num_maps, num_som_neurons_per_map,
                    # som_neuron_weight_vector_size]
                    # Output Tensor Shape local_weights: [num_maps, num_som_neuron, num_som_neurons_prev_layer or
                    # patch_size]
                    distances_block_batch, neighborhood_coeff_block_batch, bmus_batch, input_patches_batch, \
                    som_weights, local_weights = layers.csnnLayer(input_batch,
                                        self.__layers[i]["somGrid"],
                                        self.__layers[i]["somKernel"],
                                        self.__layers[i]["strides"],
                                        train_interval=self.__layers[i]["trainInterval"],
                                        padding=self.__layers[i]["padding"],
                                        bmu_metrics=self.__layers[i]["bmuMetrics"],
                                        local_weights_type=self.__layers[i]["localWeightsType"],
                                        local_neighborhood_coeff_type=self.__layers[i]["localNeighborhoodCoeffType"],
                                        som_learning_rule=self.__layers[i]["somLearningRule"],
                                        local_learning_rule=self.__layers[i]["localLearningRule"],
                                        som_learning_rate=self.__layers[i]["somLearningRate"],
                                        local_learning_rate=self.__layers[i]["localLearningRate"],
                                        neighborhood_function=self.__layers[i]["neighborhoodFunction"],
                                        name="som2DGridLayerMasked"+str(i),
                                        verbose=self.__layers[i]["verbose"],
                                        std_coeff=self.__layers[i]["neighborhoodStdCoeff"],
                                        decrease_neighborhood_range=decrease_neighborhood_range)

                    # Create logs for Tensorboard
                    self.__tensorboardLogs(input_batch, distances_block_batch, som_weights, local_weights, i,
                                           bmus_batch, neighborhood_coeff_block_batch)

                    # Save or add output for skip connections.
                    if self.__layers[i]["skipOut"]:
                        skip_out = distances_block_batch
                    if self.__layers[i]["skipIn"]:
                        distances_block_batch = tf.add(distances_block_batch, skip_out) #TODO: Test mul

                    # Choose the normalization of the layer
                    distances_block_batch = layers.chooseNormalization(distances_block_batch,
                                                                       self.__layers[i]["bmuMetricsValueInvert"],
                                                                       self.__layers[i]["bmuMetricsNormalization"])

                    # Choose the pooling type(s) of the layer.
                    distances_block_batch = layers.choosePoolingLayer(distances_block_batch, self.__layers[i])

                    # Add dropout if wanted.
                    if self.__layers[i]["dropout"]:
                        distances_block_batch = tf.layers.dropout(distances_block_batch,
                                                                rate=self.__layers[i]["dropout"], training=is_training)

                    # The input batch for the next layer is this layers output.
                    input_batch = distances_block_batch

                    # Collect all outputs for visualisations.
                    distances_per_layer.append(distances_block_batch)

                    # Save the first "input" patches of the CSNN model for visualisations.
                    if i == 0:
                        input_patches_batch_out = input_patches_batch

                    # Print the convolution outputs for easier model construction.
                    print(distances_block_batch)

            return distances_per_layer, input_patches_batch_out

    def __tensorboardLogs(self, input_batch, distances_block_batch, som_weights, local_weights, layer_num, bmus,
                          neighborhood_coeff_block_batch):
        """
        Defines several logs for the Tensorboard layerwise.
        :param input_batch: (Tensor) The input of the csnn.
        :param distances_block_batch: (Tensor) The distance output of the csnn layer.
        :param som_weights: (Tensor) The som_weights of the csnn layer.
        :param local_weights: (Tensor) The local_weights of the csnn layer.
        :param layer_num: (Integer) The layer_num of the csnn layer.
        :param bmus: (Tensor) The bmus of the csnn layer.
        :param neighborhood_coeff_block_batch: (Tensor) The neighborhood_coeffs of the csnn layer.
        """

        layer_config = self.__layers[layer_num]
        if layer_config["verbose"]:
            if layer_num is 0:
                tf.summary.image("InputImage", input_batch, max_outputs=1)
            tf.summary.histogram("Layer" + str(layer_num) + "Distances", distances_block_batch)
            _, input_height, input_width, input_depth = input_batch.get_shape()
            _, num_conv_patches_h, num_conv_patches_w, _ = distances_block_batch.get_shape()

            for m in range(layer_config["somGrid"][2]):
                visualisation.csnnWeightGridVisualisation(som_weights[m], layer_config["somGrid"],
                                                          layer_config["somKernel"][0], layer_config["somKernel"][1],
                                                          input_depth=input_depth.value, input_channel_indices=[0],
                                                          rgb_vis=self.__rgb_vis)
            if layer_num > 0:
                prev_grid = self.__model_config["layers"][layer_num - 1]["somGrid"]

            else:
                prev_grid = copy.copy((layer_config["somKernel"]))
                prev_grid.append(1)
            if local_weights:
                mask_type = layer_config["localWeightsType"]
                visualisation.csnnNeuronLocalWeightVisualisation(local_weights, [0, 3], prev_grid, mask_type)
                visualisation.csnnNeuronLocalWeightVisualisation(local_weights, [0, 25], prev_grid, mask_type)
                visualisation.csnnNeuronLocalWeightVisualisation(local_weights, [0, 50], prev_grid, mask_type)
                visualisation.csnnNeuronLocalWeightVisualisation(local_weights, [0, 75], prev_grid, mask_type)
                visualisation.csnnNeuronLocalWeightVisualisation(local_weights, [0, 100], prev_grid, mask_type)
                visualisation.csnnNeuronLocalWeightVisualisation(local_weights, [0, 144], prev_grid, mask_type)

            visualisation.csnnWeightImageVisualisation(som_weights, bmus, layer_config["somGrid"],
                                                       layer_config["somKernel"], num_conv_patches_h, num_conv_patches_w
                                                      ,map_index=0, input_channel_indices=[0], rgb_vis=self.__rgb_vis)

            visualisation.csnnSpatialActivationVisualisation(neighborhood_coeff_block_batch, [[0, 0], [3, 4], [6, 6]],
                                                             layer_config["somGrid"], name_prefix="Neighborhood")

            if layer_num == (len(self.__layers) - 1):
                d_img = tf.reshape(input_batch, [-1, layer_config["somGrid"][2], layer_config["somGrid"][0],
                                               layer_config["somGrid"][1]])
                tf.summary.image("Encoding", tf.transpose(d_img, [0, 2, 3, 1]), max_outputs=1)


    def getTrainOp(self, train_batch):
        """
        Constructs the operation to get the training for the Tensorflow graph.
        :param train_batch: (Tensor) The input batch tensor.
        :return: last_layer_distances_op: (Tensor) The distances (representation) of the last layer.
        """
        distances_per_layer, _ = self.__csnnFn(train_batch, is_training=True)
        last_layer_distances_op = distances_per_layer[-1]
        return last_layer_distances_op

    def getValOp(self, val_batch):
        """
        Constructs the operation to get the validation for the Tensorflow graph.
        :param val_batch: (Tensor) The input batch tensor.
        :return: distances_per_layer_ops: (Tensor) The distances (representation) for each layer.
        :return: image_patches_op: (Tensor) The image patches of the first layer.
        """
        distances_per_layer_ops, image_patches_op = self.__csnnFn(val_batch, is_training=False)
        return distances_per_layer_ops, image_patches_op

    def getInferOp(self, infer_batch, is_training=False):
        """
        Constructs the operation to get the infer operation of the Tensorflow graph.
        :param infer_batch: (Tensor) The input batch tensor.
        :param is_training: (Boolean) If true, the training graph is constructed, if false the validation graph.
        :return: last_layer_distances_op: (Tensor) The distances (representation) of the last layer.
        """
        distances_per_layer, _ = self.__csnnFn(infer_batch, is_training=is_training)
        last_layer_distances_op = distances_per_layer[-1]
        return last_layer_distances_op

    def applySomLearning(self, global_step, num_gpus):
        """
        Constructs the optimizer operations for the Tensorflow graph.
        :param: global_step: (Tensor) The global step of the model/training.
        :param: num_gpus: (Integer) The number of GPUs in use.
        :return: optimize_csnn_ops: (Tensors) The optimization operation to train all som and mask weights of each
                                     layer of the CSNN.
        """
        optimize_csnn_ops = optimize_csnn(global_step, num_gpus, self.__layer_train_intervals)
        return optimize_csnn_ops
