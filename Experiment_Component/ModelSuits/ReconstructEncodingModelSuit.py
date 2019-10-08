import time
import os
import shutil

import numpy as np
import tensorflow as tf


from Experiment_Component.AModelSuit import AModelSuit

class ReconstructEncodingModelSuit(AModelSuit):
    """
    A StandardTfRecordModelSuit handles the train/eval/test/inference of simple Deep Learning models like MLPs or CNNs.
    Therefore it brings the DatasetProvider, the model and the trainer, together in one place.

    It defines training validation and inference loops/operations for the model.

    :Attributes
        __train_summary_op: (tf.summary) The summary of the training for Tensorboard.
    """

    def __init__(self, model, dataset, batch_size, trainer, num_gpus, x_shape=[None, 32, 32, 3],
                 y_shape=[None, 32, 32, 3], model_dir="/model", save_checkpoint_interval=500, log_interval=100,
                 save_summary_interval=250):
        """
        Constructor, initialize member variables.
        :param model: ("Model") The model to handle with the ModelSuit
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
        # Set number of GPUs.
        self.__num_gpus = num_gpus

        # Start and init a Session.
        sess = tf.Session()

        # Preparing dataset.
        self._dataset = dataset
        self._placeholder_x = tf.placeholder(tf.float32, x_shape)
        self._placeholder_y = tf.placeholder(tf.float32, y_shape)

        # Build training_ops via trainer.
        self._training_ops = trainer.createTraining(model, self._placeholder_x, self._placeholder_y)
        self._val_ops = trainer.createValidation(model, self._placeholder_x, self._placeholder_y)

        super().__init__(sess, model, batch_size, trainer, model_dir=model_dir,
                         save_checkpoint_interval=save_checkpoint_interval, log_interval=log_interval,
                         save_summary_interval=save_summary_interval)

        # To check the parameters of the model
        self.calcNumTrainableParams()

        # Get the models train summary ops.
        tf.summary.scalar('loss', self._training_ops[1])
        self.__train_summary_op = tf.summary.merge_all()


    def doTraining(self, train_steps, eval_interval, only_save_best_checkpoints=True):
        """
        Trains the model with the trainer and the input of the ModelSuit.
        :param train_steps: (Integer) The steps to train the model.
        :param eval_interval: (Integer) Every eval_interval steps the model will be evaluated.
        :param only_save_best_checkpoints: (Boolean) If true only the best model checkpoints on the evaluation set will
                                            be saved. True by default.
        """
        self._logger.train(
            "Started training for " + str(train_steps) + " steps. Evaluation every " + str(eval_interval) + " steps...",
            "ReconstructEncodingModelSuit:doTraining")

        start_training_time = time.time()
        start_log_loss_steps_time = time.time()

        # Get subdataset.
        x = self._dataset["x_train"]
        y = self._dataset["i_train"]

        index = 0
        loss_value = 0
        best_loss = 999999
        try:
            # As long as the given train_steps are not reached.
            while self._global_step < train_steps:
                if (index + self._batch_size) < len(y):
                    new_index = index + self._batch_size
                else:
                    new_index = len(y)

                encoding_batch = x[index:new_index]
                label_batch = y[index:new_index]

                # Run model training operations.
                _, loss_value, _, summary = self._sess.run((self._training_ops[0], self._training_ops[1],
                                                                    self._training_ops[2], self.__train_summary_op),
                                                                    feed_dict={self._placeholder_x: encoding_batch,
                                                                               self._placeholder_y: label_batch})

                # If log_interval steps past print the logs.
                if self._global_step % self._log_interval == 0:
                    end_log_loss_steps_time = time.time()
                    self._logger.train("Step " + str(self._global_step) + ": " + str(self._log_interval) +
                                        " steps past in " + str(end_log_loss_steps_time-start_log_loss_steps_time) +
                                        "s. Loss value: " + str(loss_value),
                                        "ReconstructEncodingModelSuit:doTraining", self._global_step, train_steps)
                    start_log_loss_steps_time = time.time()

                # If evaluation is wished.
                if eval_interval != 0:
                    # And if eval_interval steps past do validation.
                    if self._global_step % eval_interval == 0:
                        eval_loss = self.doValidation("eval")
                        # And if only_save_best_checkpoints is set and the eval_acc is higher then the best save model.
                        if only_save_best_checkpoints and (best_loss > eval_loss):
                            self._saver.save(self._sess, self._model_dir + '/checkpoints/model',
                                                global_step=self._global_step,
                                                write_meta_graph=False)
                            best_loss = eval_loss

                # If a summary should be saved and save_summary_interval steps past the summary.
                if (self._save_summary_interval != 0) and (self._global_step % self._save_summary_interval == 0):
                    self._summary_writer.writeTrainSummary(summary, self._global_step)

                # Save checkpoint every save_checkpoint_interval iterations if only_save_best_checkpoints is not set.
                if (not only_save_best_checkpoints) and (self._global_step % self._save_checkpoint_interval == 0):
                    self._saver.save(self._sess, self._model_dir + '/checkpoints/model', global_step=self._global_step,
                                         write_meta_graph=False)

                # Increment the global step.
                self._global_step += 1

                if index >= len(y):
                    p = np.random.RandomState().permutation(len(y))
                    x = x[p]
                    y = y[p]
                    index = 0
                else:
                    index = new_index

                if self._global_step >= train_steps:
                    break

            # Stop training time.
            end_training_time = time.time()

            self._logger.train("Finished training for " + str(train_steps) + " steps. Evaluation was every " +
                               str(eval_interval) + " steps. Training duration was: " +
                               str(end_training_time-start_training_time) + "s. Final loss value: " + str(loss_value),
                               "ReconstructEncodingModelSuit:doTraining")

            self.doValidation("test")

        except tf.errors.OutOfRangeError:
            self._logger.warning(
                "Error in step: " + str(self._global_step)+" before the training finished!",
                "ReconstructEncodingModelSuit:doTraining")

    def doValidation(self, mode):
        """
        Validates the model on the subdataset subset defined by the mode.
        :param mode: (String) The subset of the dataset ("train", "eval" or "test).
        :return: loss: (Float) The loss of the validation.
        """
        self._logger.val("Started validation for " + str(mode) + "...", "ReconstructEncodingModelSuit:doValidation")

        start_validation_time = time.time()
        start_log_loss_steps_time = time.time()

        # Get subdataset.
        x = self._dataset["x_" + str(mode)]
        y = self._dataset["i_" + str(mode)]

        index = 0
        val_step = 0
        loss_values = []
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
                loss_value, _ = self._sess.run((self._val_ops[0], self._val_ops[1]),
                                                       feed_dict={self._placeholder_x: encoding_batch,
                                                                  self._placeholder_y: label_batch})

                # If log_interval steps past print the logs.
                if (val_step % self._log_interval == 0) and (val_step != 0):
                    end_log_loss_steps_time = time.time()
                    self._logger.val("Step " + str(self._global_step) + ", " + str(mode) + " step " + str(val_step) +
                                     ": " + str(self._log_interval) + " " + str(mode) + " steps past in " +
                        str(end_log_loss_steps_time - start_log_loss_steps_time) + "s. Loss value: " + str(loss_value),
                                     "ReconstructEncodingModelSuit:doValidation")
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
                "ReconstructEncodingModelSuit:doValidation")

        self._logger.val("Validation complete. Iterator went out of range as wished in validation step: " +
                         str(val_step), "ReconstructEncodingModelSuit:doValidation")

        # Calculate the validation results of the dataset.
        loss = np.mean(loss_values)

        # Log the validation results of the dataset in Tensorboard.
        summary = tf.Summary()
        summary.value.add(tag="loss", simple_value=loss)
        self._summary_writer.writeSummary(summary, self._global_step, mode)
        end_validation_time = time.time()

        # Print the validation results.
        self._logger.val("Finished " + str(mode) + " for " + str(val_step) + " steps. Validation duration was: " +
                str(end_validation_time-start_validation_time) + "s. Final loss value: " + str(loss),
                "StandardTfRecordModelSuit:doValidation")

        # Log the validation results in a text file.
        self._summary_txt_writer.writeSummary("Loss for step " + str(val_step) + ": " + str(loss), mode)
        return loss


    def saveReconstructions(self, mode):
        """
        Saves the reconstructions of the given mode every save_steps steps to file.
        :param mode: (String) The subset of the dataset ("train", "eval" or "test).
        """
        self._logger.val("Started saveReconstructions for " + str(mode) + "...",
                         "ReconstructEncodingModelSuit:saveReconstructions")

        start_log_loss_steps_time = time.time()

        # Get subdataset.
        x = self._dataset["x_" + str(mode)]
        l = self._dataset["y_" + str(mode)]
        y = self._dataset["i_" + str(mode)]

        index = 0
        val_step = 0
        recs = []
        labels = []
        images = []
        try:
            # Run until the end of the subdataset.
            while index < len(y):
                # Index to go over the entire subdataset.
                if (index + self._batch_size) < len(y):
                    new_index = index + self._batch_size
                else:
                    new_index = len(y)

                encoding_batch = x[index:new_index]
                label_batch = y[index:new_index]
                label_name_batch = l[index:new_index]

                # Run model validation operations.
                loss_value, rec = self._sess.run((self._val_ops[0], self._val_ops[1]),
                                               feed_dict={self._placeholder_x: encoding_batch,
                                                          self._placeholder_y: label_batch})
                for i in range(0, rec.shape[0]):
                    recs.append(rec[i])
                    labels.append(label_name_batch[i])
                    images.append(label_batch[i])

                # If log_interval steps past print the logs.
                if (val_step % self._log_interval == 0) and (val_step != 0):
                    end_log_loss_steps_time = time.time()
                    self._logger.val("Step " + str(self._global_step) + ", " + str(mode) + " step " + str(val_step) +
                                     ": " + str(self._log_interval) + " " + str(mode) + " steps past in " +
                                     str(end_log_loss_steps_time - start_log_loss_steps_time) + "s. Loss value: " + str(
                        loss_value),
                                     "ReconstructEncodingModelSuit:saveReconstructions")
                    start_log_loss_steps_time = time.time()

                # Increment the val_step.
                index = new_index
                val_step += 1
                if index >= len(y):
                    break

        except tf.errors.OutOfRangeError:
            self._logger.warning(
                "Error in step: " + str(self._global_step) + " before the saveReconstructions finished!",
                "ReconstructEncodingModelSuit:doValidation")

        if os.path.exists(self._model_dir + "/reconstructions/"):
            shutil.rmtree(self._model_dir + "/reconstructions/")
        os.makedirs(self._model_dir + "/reconstructions/")

        np.save(self._model_dir + "/reconstructions/" + str(mode) + "_recs", np.array(recs))
        np.save(self._model_dir + "/reconstructions/" + str(mode) + "_labels", np.array(labels))
        np.save(self._model_dir + "/reconstructions/" + str(mode) + "_images", np.array(images))

        self._logger.val("Finished saveReconstructions for " + str(mode),
                         "ReconstructEncodingModelSuit:saveReconstructions")





