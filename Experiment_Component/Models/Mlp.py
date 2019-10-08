import tensorflow as tf

class Mlp():
    """
    Configurable MLP model in plain Tensorflow.

    Hyperparameters to configure:
        adam learning rate.
        number of classes of the output layer
        number of layers
        per layer:
            neurons per layer
            dropout rate after each hidden layer

    :Attributes:
        __model_config: (Dictionary) The configuration of the MLP.
    """
    def __init__(self, model_config):
        """
        Constructor, initialize member variables.
        :param model_config: (Dictionary) The configuration of the MLP.
        """
        self.__model_config = model_config

    def __mlpNetFn(self, input_batch, is_training):
        """
        Constructs the core MLP Tensorflow graph.
        :param input_batch: (Tensor) The input batch tensor.
        :param is_training: (Boolean) If true, the training graph is constructed, if false the validation graph.
        :return: logits: (Graph) The numClasses predicted logits of the MLP.
        """
        with tf.variable_scope('Mlp'):
            x = tf.layers.flatten(input_batch)
            for layer in self.__model_config["layers"]:
                x = tf.layers.dense(x, layer["neurons"], activation=layer["activation"])
                x = tf.layers.dropout(x, rate=layer["dropoutRate"], training=is_training)

            # Output layer, class prediction.
            logits = tf.layers.dense(x, self.__model_config["numClasses"])
        return logits

    def getTrainLossOp(self, train_batch, train_labels):
        """
        Constructs the operation to get the training loss and logits for the Tensorflow graph.
        :param train_batch: (Tensor) The input batch tensor.
        :param train_labels: (Tensor) The label tensor.
        :return: train_loss_op: (Tensor) The loss of the predictions of the MLP.
        :return: train_logits_op: (Tensor) The numClasses predicted logits of the MLP.
        """
        train_logits_op = self.__mlpNetFn(train_batch, is_training=True)

        train_loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=train_logits_op, labels=tf.cast(train_labels, dtype=tf.int32)))

        return train_loss_op, train_logits_op

    def getValLossOp(self, val_batch, val_labels):
        """
        Constructs the operation to get the validation loss and logits for the Tensorflow graph.
        :param val_batch: (Tensor) The input batch tensor.
        :param val_labels: (Tensor) The label tensor.
        :return: val_loss_op: (Tensor) The loss of the predictions of the MLP.
        :return: val_logits_op: (Tensor) The numClasses predicted logits of the MLP.
        """
        val_logits_op = self.__mlpNetFn(val_batch, is_training=False)

        val_loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=val_logits_op, labels=tf.cast(val_labels, dtype=tf.int32)))

        return val_loss_op, val_logits_op

    def getEvaluationOp(self, eval_logits, eval_labels):
        """
        Constructs the operation to evaluate the given logits on the given labels for the Tensorflow graph.
        :param eval_logits: (Tensor) The logits to evaluate.
        :param eval_labels: (Tensor) The label to evaluate on.
        :return: eval_acc_op: (Tensor) The accuracy of the logits.
        """
        eval_logits = tf.reshape(eval_logits, [-1, self.__model_config["numClasses"]])
        correct_prediction = tf.equal(tf.argmax(eval_logits, 1), tf.argmax(eval_labels, 1))
        eval_acc_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return eval_acc_op

    def getInferOp(self, infer_batch):
        """
        Constructs the inference operation for the Tensorflow graph.
        :param infer_batch: (Tensor) The input batch tensor.
        :return: infer_op: (Tensor) The inference operation.
        """
        infer_op = self.__mlpNetFn(infer_batch, is_training=False)
        return infer_op

    def getOptimizerOp(self, global_step):
        """
        Constructs the optimizer operation for the Tensorflow graph.
        :param: global_step: (Tensor) The global step of the model/training.
        :return: optimizer_op: (tf.train.AdamOptimizer) The optimizer to train the model.
        """
        optimizer_op = tf.train.AdamOptimizer(learning_rate=self.__model_config["learningRate"])
        return optimizer_op


