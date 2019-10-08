import tensorflow as tf

class Cifar10Reconstruction():
    """
    Cifar10 reconstruction model.

    Hyperparameters to configure:
        adam learning rate.

    :Attributes:
        __model_config: (Dictionary) The configuration of the MLP.
    """
    def __init__(self, model_config):
        """
        Constructor, initialize member variables.
        :param model_config: (Dictionary) The configuration of the MLP.
        """
        self.__model_config = model_config

    def __cnnNetFn(self, input_batch, is_training):
        """
        Constructs the core Tensorflow graph.
        :param input_batch: (Tensor) The input batch tensor.
        :param is_training: (Boolean) If true, the training graph is constructed, if false the validation graph.
        :return: rec: (Graph) The reconstruction.
        """
        with tf.variable_scope('cnn'):

            x = tf.layers.conv2d(input_batch, 512, 3, padding='SAME')
            x = tf.layers.batch_normalization(x)
            x = tf.nn.relu(x)
            x = tf.image.resize_images(x, size=(8, 8), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

            x = tf.layers.conv2d(x, 256, 3, padding='SAME')
            x = tf.layers.batch_normalization(x)
            x = tf.nn.relu(x)
            x = tf.image.resize_images(x, size=(16, 16), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

            x = tf.layers.conv2d(x, 128, 3, padding='SAME')
            x = tf.layers.batch_normalization(x)
            x = tf.nn.relu(x)
            x = tf.image.resize_images(x, size=(32, 32), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

            x = tf.layers.conv2d(x, 64, 3, padding='SAME')
            x = tf.layers.batch_normalization(x)
            x = tf.nn.relu(x)

            x = tf.layers.conv2d(x, 3, 3, padding='SAME')
            x = tf.layers.batch_normalization(x)
            rec = tf.sigmoid(x)

        return rec

    def getTrainLossOp(self, train_batch, train_labels):
        """
        Constructs the operation to get the training loss and logits for the Tensorflow graph.
        :param train_batch: (Tensor) The input batch tensor.
        :param train_labels: (Tensor) The label tensor.
        :return: train_loss_op: (Tensor) The loss of the reconstructions.
        :return: train_logits_op: (Tensor) The reconstruction.
        """
        train_logits_op = self.__cnnNetFn(train_batch, is_training=True)

        train_loss_op = tf.reduce_mean(tf.losses.mean_squared_error(labels=train_labels, predictions=train_logits_op))

        if self.__model_config["verbose"]:
            tf.summary.image("train_rec", train_logits_op, max_outputs=10)
            tf.summary.image("train_label", train_labels, max_outputs=10)

        return train_loss_op, train_logits_op

    def getValLossOp(self, val_batch, val_labels):
        """
        Constructs the operation to get the validation loss and logits for the Tensorflow graph.
        :param val_batch: (Tensor) The input batch tensor.
        :param val_labels: (Tensor) The label tensor.
        :return: val_loss_op: (Tensor) The loss of the reconstructions.
        :return: val_logits_op: (Tensor) The reconstruction.
        """
        val_logits_op = self.__cnnNetFn(val_batch, is_training=False)

        val_loss_op = tf.reduce_mean(tf.losses.mean_squared_error(labels=val_labels, predictions=val_logits_op))

        if self.__model_config["verbose"]:
            tf.summary.image("val_rec", val_logits_op, max_outputs=10)
            tf.summary.image("val_label", val_labels, max_outputs=10)

        return val_loss_op, val_logits_op

    def getEvaluationOp(self, eval_logits, eval_labels):
        """
        Constructs the operation to evaluate the given logits on the given labels for the Tensorflow graph.
        :param eval_logits: (Tensor) The logits to evaluate.
        :param eval_labels: (Tensor) The label to evaluate on.
        :return: eval_acc_op: (Tensor) The accuracy of the logits.
        """
        return eval_logits

    def getInferOp(self, infer_batch):
        """
        Constructs the inference operation for the Tensorflow graph.
        :param infer_batch: (Tensor) The input batch tensor.
        :return: infer_op: (Tensor) The inference operation.
        """
        infer_op = self.__cnnNetFn(infer_batch, is_training=False)
        return infer_op

    def getOptimizerOp(self, global_step):
        """
        Constructs the optimizer operation for the Tensorflow graph.
        :param: global_step: (Tensor) The global step of the model/training.
        :return: optimizer_op: (tf.train.AdamOptimizer) The optimizer to train the model.
        """
        optimizer_op = tf.train.AdamOptimizer(learning_rate=self.__model_config["learningRate"])
        return optimizer_op


