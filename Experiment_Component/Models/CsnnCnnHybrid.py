import tensorflow as tf

from Experiment_Component.Models.Csnn import Csnn

class CsnnCnnHybrid():
    """
    Configurable CsnnCnnHybrid model in plain Tensorflow.

    Hyperparameters to configure:
        adam learning rate.
        number of classes of the output layer
        CSNN parameters.

    :Attributes:
        __model_config: (Dictionary) The configuration of the MLP.
    """

    def __init__(self, model_config):
        """
        Constructor, initialize member variables.
        :param model_config: (Dictionary) The configuration of the MLP.
        """
        # Training Parameters
        self.__learning_rate = model_config["cnnLearningRate"]

        # Network Parameters
        self.__num_classes = model_config["numClasses"]
        self.__weight_decay = 1e-4
        self.__num_gpus = model_config["numGpus"]
        self.__use_csnn = model_config["useCsnn"]

        self.__csnn = Csnn(model_config)

    def __cnnNetFn(self, input, is_training):
        """
        Constructs the core CSNN-CNN hyprid Tensorflow graph.
        :param input_batch: (Tensor) The input batch tensor.
        :param is_training: (Boolean) If true, the training graph is constructed, if false the validation graph.
        :return: logits: (Graph) The numClasses predicted logits of the MLP.
        :return: csnn_features: (Tensor) The distances (representation) for each layer.
        """
        with tf.variable_scope('CNN'):
            conv1 = tf.layers.conv2d(input, 32, 3, activation="elu", padding='SAME',
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(self.__weight_decay))
            conv1_bn = tf.layers.batch_normalization(conv1)
            conv2 = tf.layers.conv2d(conv1_bn, 32, 3, activation="elu", padding='SAME',
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(self.__weight_decay))
            conv2_bn = tf.layers.batch_normalization(conv2)
            conv2_pool = tf.layers.max_pooling2d(conv2_bn, 2, 2, padding='SAME')
            conv2_drop = tf.layers.dropout(conv2_pool, rate=0.2, training=is_training)

            conv3 = tf.layers.conv2d(conv2_drop, 64, 3, activation="elu", padding='SAME',
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(self.__weight_decay))
            conv3_bn = tf.layers.batch_normalization(conv3)
            conv4 = tf.layers.conv2d(conv3_bn, 64, 3, activation="elu", padding='SAME',
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(self.__weight_decay))
            conv4_bn = tf.layers.batch_normalization(conv4)
            conv4_pool = tf.layers.max_pooling2d(conv4_bn, 2, 2, padding='SAME')
            conv4_drop = tf.layers.dropout(conv4_pool, rate=0.3, training=is_training)

            conv5 = tf.layers.conv2d(conv4_drop, 128, 3, activation="elu", padding='SAME',
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(self.__weight_decay))
            conv5_bn = tf.layers.batch_normalization(conv5)
            conv6 = tf.layers.conv2d(conv5_bn, 128, 3, activation="elu", padding='SAME',
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(self.__weight_decay))
            conv6_pool = tf.layers.max_pooling2d(conv6, 2, 2, padding='SAME')

            csnn_features = tf.stop_gradient(self.__csnn.getTrainOp(input))
            csnn_features = tf.identity(csnn_features)
            if self.__use_csnn:
                joint_features = tf.concat((conv6_pool, csnn_features), axis=3)
            else:
                joint_features = conv6_pool

            conv6_bn = tf.layers.batch_normalization(joint_features)

            conv7 = tf.layers.conv2d(conv6_bn, 256, 3, activation="elu", padding='SAME',
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(self.__weight_decay))
            conv7_bn = tf.layers.batch_normalization(conv7)
            conv8 = tf.layers.conv2d(conv7_bn, 256, 3, activation="elu", padding='SAME',
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(self.__weight_decay))
            conv8_bn = tf.layers.batch_normalization(conv8)
            conv8_pool = tf.layers.max_pooling2d(conv8_bn, 2, 2, padding='SAME')
            conv8_drop = tf.layers.dropout(conv8_pool, rate=0.4, training=is_training)

            flat = tf.contrib.layers.flatten(conv8_drop)
            logits = tf.layers.dense(flat, self.__num_classes)
        return logits, csnn_features

    def getTrainLossOp(self, train_images, train_labels):
        """
        Constructs the operation to get the training loss logits, csnn_distances for the Tensorflow graph.
        :param train_batch: (Tensor) The input batch tensor.
        :param train_labels: (Tensor) The label tensor.
        :return: loss_op: (Tensor) The loss of the predictions of the MLP.
        :return: logits_train: (Tensor) The numClasses predicted logits of the MLP.
        :return: csnn_distances: (Tensor) The distances (representation) of the last layer.
        """
        logits_train, csnn_distances = self.__cnnNetFn(train_images, is_training=True)
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=logits_train, labels=tf.cast(train_labels, dtype=tf.int32)))

        return loss_op, logits_train, csnn_distances

    def getValLossOp(self, eval_images, eval_labels):
        """
        Constructs the operation to get the validation loss logits, csnn_distances for the Tensorflow graph.
        :param eval_images: (Tensor) The input batch tensor.
        :param eval_labels: (Tensor) The label tensor.
        :return: loss_op: (Tensor) The loss of the predictions of the MLP.
        :return: logits_eval: (Tensor) The numClasses predicted logits of the MLP.
        :return: csnn_distances: (Tensor) The distances (representation) of the last layer.
        """
        logits_eval, csnn_distances = self.__cnnNetFn(eval_images, is_training=False)
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=logits_eval, labels=tf.cast(eval_labels, dtype=tf.int32)))
        return loss_op, logits_eval, csnn_distances

    def getEvaluationOp(self, logits, labels):
        """
        Constructs the operation to evaluate the given logits on the given labels for the Tensorflow graph.
        :param logits: (Tensor) The logits to evaluate.
        :param correct_prediction: (Tensor) The label to evaluate on.
        :return: acc_op: (Tensor) The accuracy of the logits.
        """
        logits = tf.reshape(logits, [-1, self.__num_classes])
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        acc_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return acc_op

    def getCnnOptimizerOp(self, global_step):
        """
        Constructs the optimizer operation for the Tensorflow graph.
        :param: global_step: (Tensor) The global step of the model/training.
        :return: optimizer_op: (tf.train.AdamOptimizer) The optimizer to train the model.
        """
        # Define loss and optimizer
        optimizer = tf.train.AdamOptimizer(self.__learning_rate)
        return optimizer

    def getCsnnOptimizerOp(self, global_step):
        """
        Constructs the csnn optimizer operations for the Tensorflow graph.
        :param: global_step: (Tensor) The global step of the model/training.
        :return: csnn_optimizer: (Tensors) The optimization operation to train all som and mask weights of each
                                     layer of the CSNN.
        """
        # Define loss and optimizer
        csnn_optimizer, _ = self.__csnn.applySomLearning(global_step, self.__num_gpus)
        return csnn_optimizer

