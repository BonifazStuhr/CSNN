import time
import tensorflow as tf
import numpy as np

from Experiment_Component.AModelSuit import AModelSuit

class AGeneratorModelSuit(AModelSuit):
    """
    A AGeneratorModelSuit handles the train/eval/test/inference of a model trained with a Keras dataset Generator.
    Therefore it brings the Generator, the model and the trainer, together in one place.
    In each AGeneratorModelSuit functions for the training and validation must be defined.

    The AGeneratorModelSuit provides basic functionality like a standard training loop and inference operation.

    :Attributes:
        __num_gpus:            (Integer) The number of GPUs to use.
        _dataset_generator:    (KerasGenerator) The Generator to generate te dataset with (for training).
        _dataset:              (Dictionary) The dataset e.g. {"x_train":(train_size, 3, 32, 32),
                               "y_train":(train_size,) or if onehot (train_size, 10), x_eval....
        _placeholder_x:        (tf.placeholder) The placeholder for the input data coming from the Generator.
        _placeholder_y:        (tf.placeholder) The placeholder for the labels coming from the Generator.
        _training_ops:         (Tensor) The training operation fpr the model created from the ITrainer.
        _val_ops:              (Tensor) The validation operation fpr the model created from the ITrainer.
        _infer_ops:            (Tensor) The validation operation fpr the model created from the ModelSuit.
                               Tensorboard summaries.
        _eval_summary_op:      (tf.summary) The summary of the evaluation for Tensorboard.
        _train_summary_op:     (tf.summary) The summary of the training for Tensorboard.
    """

    def __init__(self, model, dataset_generator, dataset, batch_size, trainer, num_gpus, x_shape=[None, 32, 32, 3],
                 y_shape=[None, 10], model_dir="/model", save_checkpoint_interval=500, log_interval=100,
                 save_summary_interval=250):
        """
        Constructor, initialize member variables.
        :param model: ("Model") The model to handle with the ModelSuit
        :param dataset_generator: KerasGenerator) The Generator to generate te dataset with (for training).
        :param dataset: (Dictionary) The dataset e.g. {"x_train":(train_size, 3, 32, 32),
        :param batch_size: (Integer) The batch size for the model.
        :param trainer: (ITrainer) The trainer to train the model.
        :param num_gpus: (Integer) The number of GPUs to use.
        :param x_shape: (Array) The shape of the data.
        :param y_shape: (Array) The shape of the labels.
        :param model_dir: (String) The directory of the model (e.g. to save it). "/model" by default.
        :param save_checkpoint_interval: (Integer) Every _save_checkpoint_interval steps the ModelSuit saves model
                                        (training) checkpoints. 500 by default.
        :param log_interval: (Integer) Every log_interval steps the ModelSuit writes logs. 100 by default.
        :param save_summary_interval: (Integer) Every save_summary_interval steps the ModelSuit saves Tensorboard
                                       summaries. 250 by default.
        """
        # Start and init a Session.
        sess = tf.Session()

        # Set number of GPUs.
        self._num_gpus = num_gpus

        # Preparing dataset.
        self._dataset_generator = dataset_generator
        self._dataset = dataset
        self._placeholder_x = tf.placeholder(tf.float32, x_shape)
        self._placeholder_y = tf.placeholder(tf.float32, y_shape)

        # Create graph operations.
        self._training_ops = trainer.createTraining(model, self._placeholder_x,  self._placeholder_y)
        self._val_ops = trainer.createValidation(model, self._placeholder_x, self._placeholder_y)
        self._infer_ops = self.createInferOps(model, batch_size, is_training=False)

        super().__init__(sess, model, batch_size, trainer, model_dir=model_dir,
                         save_checkpoint_interval=save_checkpoint_interval, log_interval=log_interval,
                         save_summary_interval=save_summary_interval)

        # To save summary.
        self._eval_summary_op = tf.summary.merge_all()
        self._train_summary_op = tf.summary.merge_all()

        # To check the parameters of the model
        self.calcNumTrainableParams()

    def doTraining(self, train_steps, eval_interval, only_save_best_checkpoints=None):
        """
        Trains the model with the trainer and the input of the ModelSuit.
        :param train_steps: (Integer) The steps to train the model.
        :param eval_interval: (Integer) Every eval_interval steps the Model will be evaluated.
        :param only_save_best_checkpoints: (Boolean) If true only the best Model checkpoints on the evaluation set will
                                            be saved. Not used.
        """
        self._logger.train(
            "Started training for " + str(train_steps) + " steps. Evaluation every " + str(eval_interval) + " steps...",
            "AGeneratorModelSuit:doTraining")

        # Stop times.
        start_training_time = time.time()
        start_log_loss_steps_time = time.time()
        self._txt_function_time_stopper.startNewFunction("doTraining")

        loss_value = 0
        acc_value = 0
        best_acc = 0
        try:
            while self._global_step < train_steps:

                for x_batch, y_batch in self._dataset_generator.flow(self._dataset["x_train"], self._dataset["y_train"],
                                                                     batch_size=self._batch_size, seed=42):
                    # Metadata containing additional graph information like execution times of various parts.
                    run_metadata = None

                    # If a summary should be saved and save_summary_interval steps past run train and summary ops.
                    if (self._save_summary_interval != 0) and (self._global_step % self._save_summary_interval == 0):
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()
                        loss_value, acc_value, _, summary = self._sess.run((self._training_ops[0], self._training_ops[1],
                                                           self._training_ops[2], self._train_summary_op),
                                                           feed_dict={self._placeholder_x: x_batch,
                                                                    self._placeholder_y: y_batch},
                                                           options=run_options, run_metadata=run_metadata)
                    # Else just run train operations.
                    else:
                       loss_value, acc_value, _ = self._sess.run((self._training_ops[0], self._training_ops[1],
                                                                  self._training_ops[2]),
                                                feed_dict={self._placeholder_x: x_batch, self._placeholder_y: y_batch})

                    # If log_interval steps past print the logs.
                    if self._global_step % self._log_interval == 0:
                        end_log_loss_steps_time = time.time()
                        self._logger.train("Step " + str(self._global_step) + ": " + str(self._log_interval) +
                                           " steps past in " + str(end_log_loss_steps_time-start_log_loss_steps_time) +
                                           "s. Accuracy: " + str(acc_value*100) + "%. Loss value: " + str(loss_value),
                                           "AGeneratorModelSuit:doTraining", self._global_step, train_steps)
                        start_log_loss_steps_time = time.time()

                    # If evaluation is wished.
                    if eval_interval != 0:
                        # And if eval_interval steps past do validation.
                        if self._global_step % eval_interval == 0:
                            eval_acc = self.doValidation("eval")
                            # And if only_save_best_checkpoints is set and the eval_acc is higher then the best save model.
                            if only_save_best_checkpoints and (best_acc < eval_acc):
                                self._saver.save(self._sess, self._model_dir + '/checkpoints/model',
                                                 global_step=self._global_step,
                                                 write_meta_graph=False)
                                best_acc = eval_acc

                    # If a summary should be saved and save_summary_interval steps past the summary.
                    if (self._save_summary_interval != 0) and (self._global_step % self._save_summary_interval == 0):
                        self._summary_writer.writeTrainSummary(summary, self._global_step)

                    # Save checkpoint every save_checkpoint_interval iterations if only_save_best_checkpoints is not set.
                    if (not only_save_best_checkpoints) and (self._global_step % self._save_checkpoint_interval == 0):
                        self._saver.save(self._sess, self._model_dir + '/checkpoints/model', global_step=self._global_step,
                                         write_meta_graph=False)

                    # Increment the global step.
                    self._global_step += 1
                    if self._global_step >= train_steps:
                        break

             # Save the model at the end. if only_save_best_checkpoints is not set.
            if not only_save_best_checkpoints:
                self._saver.save(self._sess, self._model_dir + '/checkpoints/model', global_step=self._global_step,
                                 write_meta_graph=False)

            # Stop training time.
            end_training_time = time.time()

            self._logger.train("Finished training for " + str(train_steps) + " steps. Evaluation was every " +
                               str(eval_interval) + " steps. Training duration was: " +
                               str(end_training_time-start_training_time) + "s. Final accuracy: " + str(acc_value*100) +
                               "%. Final loss value: " + str(loss_value), "AGeneratorModelSuit:doTraining")

            self.doValidation("test")


        except tf.errors.OutOfRangeError:
            self._logger.warning(
                "Error in step: " + str(
                    self._global_step) + " before the training finsihed!", "AGeneratorModelSuit:doTraining")


    def createInferOps(self, model, batch_size, is_training=False):
        """
        Creates the inference operations for the model on the given GPUs.
        :param model: ("Model") The model to create the inference operations for.
        :param batch_size: (Integer) The batch size for the model.
        :param is_training: (Boolean) If true the inference is created from the training part of the graph.
        :return: inference_ops: (Tensor) The operation to get the inference result.
        :return: label_ops: (Tensor) The operation to get the labels of the inference result.
        """
        with tf.variable_scope(tf.get_variable_scope()) as outer_scope:
            outer_scope.reuse_variables()
            inference_ops = []
            label_ops = []
            # If there is just one GPU or the batch size is smaller as the number of GPUs, create operation for one GPU.
            if (self._num_gpus <= 1) or (batch_size < self._num_gpus):
                with tf.name_scope('Infer'):
                    inference_ops = model.getInferOp(self._placeholder_x, is_training)
                    inference_ops = [inference_ops]
                    label_ops = [self._placeholder_y]
            # Else split the input evenly and create operations for each GPU.
            else:
                input_data_split = tf.split(self._placeholder_x, self._num_gpus)
                labels_split = tf.split(self._placeholder_y, self._num_gpus)
                for gpu in range(self._num_gpus):
                    with tf.name_scope('InferGPU%d' % gpu), tf.device('/gpu:%d' % gpu):
                        inference_ops.append(model.getInferOp(input_data_split[gpu], is_training))
                        label_ops.append(labels_split[gpu])
            return inference_ops, label_ops


    def doValidation(self, mode):
        """
        Validates the model on the subdataset subset defined by the mode.
        :param mode: (String) The subset of the dataset ("train", "eval" or "test).
        :return: loss: (Float) The loss of the validation.
        """
        self._logger.val("Started validation for " + str(mode) + "...", "AGeneratorModelSuit:doValidation")

        start_validation_time = time.time()
        start_log_loss_steps_time = time.time()

        # Get subdataset.
        x = self._dataset["x_" + str(mode)]
        y = self._dataset["y_" + str(mode)]

        index = 0
        val_step = 0
        loss_values = []
        acc_values = []
        try:
            # Run until the end of the subdataset.
            while True:
                # Index to go over the entire subdataset.
                if (index + self._batch_size) < len(y):
                    new_index = index + self._batch_size
                else:
                    new_index = len(y)

                encoding_batch = x[index:new_index]
                label_batch = y[index:new_index]

                # Run model validation operations.
                loss_value, acc_value = self._sess.run((self._val_ops[0], self._val_ops[1]),
                                               feed_dict={self._placeholder_x: encoding_batch,
                                                          self._placeholder_y: label_batch})

                # If log_interval steps past print the logs.
                if (val_step % self._log_interval == 0) and (val_step != 0):
                    end_log_loss_steps_time = time.time()
                    self._logger.val("Step " + str(self._global_step) + ", " + str(mode) + " step " + str(val_step) +
                                     ": " + str(self._log_interval) + " " + str(mode) + " steps past in " +
                                     str(end_log_loss_steps_time - start_log_loss_steps_time) + "s. Accuracy: " +
                                     str(acc_value * 100) + "%. Loss value: " + str(loss_value),
                                     "AGeneratorModelSuit:doValidation")
                    start_log_loss_steps_time = time.time()

                # Append the validation results of the batch.
                loss_values.append(loss_value)

                # Increment the val_step.
                index = new_index
                val_step += 1
                if index >= len(y):
                    break

        except tf.errors.OutOfRangeError:
            self._logger.warning(
                "Error in step: " + str(self._global_step) + " before the validation finished!",
                "AGeneratorModelSuit:doValidation")

        self._logger.val(
            "Validation complete. Iterator went out of range as wished in validation step: " + str(val_step),
            "AGeneratorModelSuit:__doValidation")

        # Calculate the validation results of the dataset.
        acc = np.mean(acc_values)
        loss = np.mean(loss_values)

        # Log the validation results of the dataset in Tensorboard.
        summary = tf.Summary()
        summary.value.add(tag="loss", simple_value=loss)
        summary.value.add(tag="acc", simple_value=acc)
        self._summary_writer.writeSummary(summary, self._global_step, mode)
        end_validation_time = time.time()

        # Print the validation results.
        self._logger.val("Finished " + str(mode) + " for " + str(val_step) + " steps. Validation duration was: " +
                         str(end_validation_time - start_validation_time) + "s. Final accuracy: " +
                         str(acc * 100) + "%. Final loss value: " + str(loss),
                         "AGeneratorModelSuit:doValidation")

        # Log the validation results in a text file.
        self._summary_txt_writer.writeSummary("Acc for step " + str(val_step) + ": " + str(acc * 100), mode)
        self._summary_txt_writer.writeSummary("Loss for step " + str(val_step) + ": " + str(loss), mode)
        return acc