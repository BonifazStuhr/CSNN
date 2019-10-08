import time
import tensorflow as tf
import numpy as np

from csnnLib import visualisation
from Experiment_Component.ModelSuits.AGeneratorModelSuit import AGeneratorModelSuit

class CsnnGeneratorModelSuit(AGeneratorModelSuit):
    """
    A CsnnGeneratorModelSuit handles the train/eval/test/inference of a CSNN model trained with a Keras dataset
    Generator. Therefore it brings the Generator, the model and the trainer, together in one place.
    It defines validation and encoding loops for the CSNN.

    :Attributes:
        __encoding_step:        (Integer) The current step of the encoding creation of the dataset.
        __log_encoding_inteval: (Integer) Every log_encoding_inteval steps the ModelSuit logs the encoding creation.
    """
    def __init__(self, model, dataset_generator, dataset, batch_size, trainer, num_gpus, x_shape=[None, 32, 32, 3],
                 y_shape=[None, 10], model_dir="/model", save_checkpoint_interval=500, log_interval=100,
                 save_summary_interval=250, log_encoding_inteval=500):
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
        :param log_encoding_inteval: (Integer) Every log_encoding_inteval steps the ModelSuit logs the encoding
                                      creation. 500 by default.
        """
        super().__init__(model, dataset_generator, dataset, batch_size, trainer, num_gpus=num_gpus, x_shape=x_shape,
                       y_shape=y_shape, model_dir=model_dir, save_checkpoint_interval=save_checkpoint_interval,
                       log_interval=log_interval, save_summary_interval=save_summary_interval)

        self.__encoding_step = 1
        self.__log_encoding_inteval = log_encoding_inteval

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
            "CsnnGeneratorModelSuit:doTraining")

        # Stop times.
        start_training_time = time.time()
        start_log_loss_steps_time = time.time()
        self._txt_function_time_stopper.startNewFunction("doTraining")

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
                        _, _, _, summary = self._sess.run((self._training_ops[0], self._training_ops[1],
                                                           self._training_ops[2], self._train_summary_op),
                                                          feed_dict={self._placeholder_x: x_batch,
                                                                     self._placeholder_y: y_batch},
                                                          options=run_options, run_metadata=run_metadata)
                    # Else just run train operations.
                    else:
                        _, _, _ = self._sess.run((self._training_ops[0], self._training_ops[1], self._training_ops[2]),
                                                 feed_dict={self._placeholder_x: x_batch, self._placeholder_y: y_batch})

                    # If log_interval steps past print the logs.
                    if self._global_step % self._log_interval == 0:
                        end_log_loss_steps_time = time.time()
                        self._logger.train(
                            "Step " + str(self._global_step) + ": " + str(self._log_interval) + " steps past in " +
                            str(end_log_loss_steps_time - start_log_loss_steps_time) + "s.",
                            "CsnnGeneratorModelSuit:doTraining",
                            self._global_step, train_steps)
                        start_log_loss_steps_time = time.time()

                    # If the model should be evaluated and eval_interval steps past evaluate the model.
                    if (eval_interval != 0) and (self._global_step != 0) and (self._global_step % eval_interval == 0):
                        self.doValidation("eval")

                    # If a summary should be saved and save_summary_interval steps past the summary.
                    if (self._save_summary_interval != 0) and (self._global_step % self._save_summary_interval == 0):
                        self._summary_writer.writeTrainSummary(summary, self._global_step, run_metadata)

                    # Save checkpoint every save_checkpoint_interval iterations
                    if self._global_step % self._save_checkpoint_interval == 0:
                        self._saver.save(self._sess, self._model_dir + '/checkpoints/model',
                                         global_step=self._global_step, write_meta_graph=False)

                    # Increment the gobal_step and finish the training if train_steps have past.
                    self._global_step += 1
                    if self._global_step >= train_steps:
                        break

            # Save the model at the end.
            self._saver.save(self._sess, self._model_dir + '/checkpoints/model', global_step=self._global_step,
                             write_meta_graph=False)

            # Stop training time.
            self._txt_function_time_stopper.stopFunctionAndWrite("doTraining")
            end_training_time = time.time()

            self._logger.train(
                "Finished training for " + str(train_steps) + " steps. Evaluation was every " + str(eval_interval) +
                " steps. Training duration was: " + str(end_training_time - start_training_time) + "s",
                "CsnnGeneratorModelSuit:doTraining")

        except tf.errors.OutOfRangeError:
            self._logger.warning(
                "Training iterator went out of range in step: " + str(
                    self._global_step) + " before the training finsihed!", "CsnnGeneratorModelSuit:doTraining")

    def createEncoding(self, mode, encoding_size=-1, return_with_encoding=None):
        """
        Creates the encoding of the subdataset. If the encoding_size is not -1 the encoding will be generated in given
        size from the Generator.
        :param mode: (String) The subset of the dataset ("train", "eval" or "test).
        :param encoding_size: (Integer) The size of the encoding to generate. -1 by default.
        :param return_with_encoding: (Dictionary) If set the encoding will be returned with given the dataset.
                                    None by default.
        :return: encoding: (Array) The generated encoding. E.g. (batch_num, gpu_num, gpu_batch_size, encoding)
        :return: encoding_labels: (Array) The labels of the encoding. E.g. (batch_num, gpu_num, gpu_batch_size, label)
        """
        if encoding_size == -1:
            return self.__createEncoding(mode, return_with_encoding)
        else:
            return self.__generateEncoding(mode, encoding_size, return_with_encoding)

    def __createEncoding(self, mode, return_with_encoding):
        """
        Creates the encoding of the subdataset with exactly the size of the dataset.
        :param mode: (String) The subset of the dataset ("train", "eval" or "test).
        :param return_with_encoding: (Dictionary) If set the encoding will be returned with given the dataset.
        :return: encoding: (Array) The generated encoding. E.g. (batch_num, gpu_num, gpu_batch_size, encoding)
        :return: encoding_labels: (Array) The labels of the encoding. E.g. (batch_num, gpu_num, gpu_batch_size, label)
        """
        self._logger.infer("Started creating encoding for " + str(mode) + "...", "CsnnModelSuit:__createEncoding")

        start_time = time.time()

        # Get subdataset.
        x = self._dataset["x_" + str(mode)]
        y = self._dataset["y_" + str(mode)]
        if return_with_encoding:
            img = return_with_encoding["x_" + str(mode)]

        index = 0
        epoch_complete = False
        encoding = []
        encoding_labels = []
        return_with = []
        while not epoch_complete:
            # Index to go over the entire subdataset.
            if (index + self._batch_size) < len(y):
                new_index = index + self._batch_size
            else:
                new_index = len(y)
                epoch_complete = True

            # Run encoding creation.
            gpu_batch_encoding, gpu_batch_labels = self._sess.run(self._infer_ops,
                                                                  feed_dict={self._placeholder_x: x[index:new_index],
                                                                  self._placeholder_y: y[index:new_index]})

            # Append generated encoding gpu_batches (num_gpu, gpu_batch_size, ...)
            encoding.append(gpu_batch_encoding)
            encoding_labels.append(gpu_batch_labels)
            if return_with_encoding:
                return_with.append(np.split(img[index:new_index], self._num_gpus))

            # If log_encoding_inteval steps past print the logs.
            if self.__encoding_step % self.__log_encoding_inteval == 0:
                self._logger.infer("Encodig step " + str(self.__encoding_step) + ": " + str(self.__log_encoding_inteval) +
                                   " encoding steps past.", "CsnnModelSuit:__createEncoding")

            # Increment dataset index and encoding step.
            index = new_index
            self.__encoding_step += 1

        end_time = time.time()

        self._logger.infer(
            "Finished create encoding for " + str(mode) + " in " + str(
                self.__encoding_step) + " steps. Encoding duration was: " +
            str(end_time - start_time) + "s.", "CsnnModelSuit:__createEncoding")

        if return_with_encoding:
            return encoding, encoding_labels, return_with

        return encoding, encoding_labels, None

    def __generateEncoding(self, mode, encoding_size, return_with_encoding):
        """
        Generates the encoding of the subdataset as long as the given encoding size is reached (more or less).
        :param mode: (String) The subset of the dataset ("train", "eval" or "test).
        :param encoding_size: (integer) The size of the encoding to generate.
        :param return_with_encoding: (Dictionary) If set the encoding will be returned with given the dataset.
        :return: encoding: (Array) The generated encoding. E.g. (batch_num, gpu_num, gpu_batch_size, encoding)
        :return: encoding_labels: (Array) The labels of the encoding. E.g. (batch_num, gpu_num, gpu_batch_size, label)
        """
        self._logger.infer("Started generating encoding for " + str(mode) + "...", "CsnnModelSuit:__generateEncoding")

        start_time = time.time()

        encoding = []
        encoding_labels = []
        return_with = []

        if return_with_encoding:
            img = return_with_encoding["x_" + str(mode)]
        index = 0

        # Use Generator to generat the dataset.
        for x_batch, y_batch in self._dataset_generator.flow(self._dataset["x_" + str(mode)],
                                                             self._dataset["y_" + str(mode)],
                                                             batch_size=self._batch_size, shuffle=False, seed=42):
            new_index = index + x_batch[0].shape[0]
            # Run encoding creation.
            gpu_batch_encoding, gpu_batch_labels = self._sess.run(self._infer_ops,
                                                                  feed_dict={self._placeholder_x: x_batch,
                                                                             self._placeholder_y: y_batch})

            # Append generated encoding gpu_batches (num_gpu, gpu_batch_size, ...)
            encoding.append(gpu_batch_encoding)
            encoding_labels.append(gpu_batch_labels)
            if return_with_encoding:
                return_with.append(np.split(img[index:new_index], self._num_gpus))

            # If log_encoding_inteval steps past print the logs.
            if self.__encoding_step % self.__log_encoding_inteval == 0:
                self._logger.infer("Encodig step " + str(self.__encoding_step) + ": " + str(self.__log_encoding_inteval)
                                   + " encoding steps past.", "CsnnModelSuit:__generateEncoding")

            # Increment encoding step and stop if the given encoding size is reached.
            self.__encoding_step += 1
            if (self.__encoding_step * self._batch_size) >= encoding_size:
                break

            if index >= len(self._dataset["y_" + str(mode)]):
                index = 0
            else:
                index = new_index

        end_time = time.time()

        self._logger.infer(
            "Finished create encoding for " + str(mode) + " in " + str(
                self.__encoding_step) + " steps. Encoding duration was: " +
            str(end_time - start_time) + "s.", "CsnnModelSuit:__generateEncoding")

        if return_with_encoding:
            return encoding, encoding_labels, return_with

        return encoding, encoding_labels, None

    def doValidation(self, mode):
        """
        Not implemented.
        Validates the model on the subdataset subset defined by the mode.
        :param mode: (String) The subset of the dataset ("train", "eval" or "test).
        """
        pass