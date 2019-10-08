import time
import numpy as np
import tensorflow as tf

from Experiment_Component.AModelSuit import AModelSuit

class FsModelSuit(AModelSuit):
    """
    A FsModelSuit handles the train/eval/test/inference of simple few shot models.
    It defines training validation and inference loops/operations for the model.

   :Attributes:
        _dataset_provider:                 (ADataProvider) The provider of the tfrecord input.
        _tfrecord_inputpipeline_config:    (Dictionary) The configuration of the input pipeline.
        _dataset_files:                    (Array) The names of the tf record files for the file_placehoders.
        _data_placeholder_train:           (tf.placeholder) The placeholder for x-shot the training data.
        _label_placeholder_train:          (tf.placeholder) TThe placeholder for x-shot the training label..
        _iterator_val:                     (tf.data.Iterator) The iterator for the validation.
        _file_placeholder_val:             (tf.placeholder)The placeholder for the dataset_files from iterator_val.
        _training_ops:                     (Tensor) The training operation fpr the model created from the ITrainer.
        _val_ops:                          (Tensor) The validation operation fpr the model created from the ITrainer.
        __train_summary_op:                (tf.summary) The summary of the training for Tensorboard.
    """

    def __init__(self, model, dataset_provider, tfrecord_inputpipeline_config, batch_size, trainer, model_dir="/model",
                 save_checkpoint_interval=500, log_interval=100, save_summary_interval=250):
        """
        Constructor, initialize member variables.
        :param model: ("Model") The model to handle with the ModelSuit
        :param dataset_provider: (ADataProvider) The provider of the tfrecord input.
        :param tfrecord_inputpipeline_config: (Dictionary) The configuration of the input pipeline.
        :param batch_size: (Integer) The batch size for the model.
        :param trainer: (ITrainer) The trainer to train the model.
        :param num_gpus: (Integer) The number of GPUs to use.
        :param model_dir: (String) The directory of the model (e.g. to save it). "/model" by default.
        :param save_checkpoint_interval: (Integer) Every _save_checkpoint_interval steps the ModelSuit saves model
                                        (training) checkpoints. 500 by default.
        :param log_interval: (Integer) Every log_interval steps the ModelSuit writes logs. 100 by default.
        :param save_summary_interval: (Integer) Every save_summary_interval steps the ModelSuit saves Tensorboard
                                       summaries. 250 by default.
        """
        # Start and init a Session.
        sess = tf.Session()

        # Preparing Pipelines.
        self._dataset_provider = dataset_provider
        self._tfrecord_inputpipeline_config = tfrecord_inputpipeline_config

        # Get the names of the input files to feed the pipeline later with train/eval/test files.
        self._dataset_files = dataset_provider.getTfRecordFileNames()

        # Build the input pipeline via the dataset provider.
        self.__shape_data_train = self._dataset_provider.getEntryShape("data")
        self.__shape_label_train = self._dataset_provider.getEntryShape("label")

        self._data_placeholder_train = tf.placeholder(self._dataset_provider.getEntryDatatyp("data"),
                                                      shape=(None, np.prod(self.__shape_data_train)))
        self._label_placeholder_train = tf.placeholder(self._dataset_provider.getEntryDatatyp("label"),
                                                      shape=(None, self.__shape_label_train[0]))

        self._iterator_val, self._file_placeholder_val, = \
            dataset_provider.getTfRecordInputPipelineIteratorsAndPlaceholdersFor("val", batch_size,
                                                                                 tfrecord_inputpipeline_config["val"])

        # Build training_ops via trainer.
        self._training_ops = trainer.createTraining(model,  self._data_placeholder_train, self._label_placeholder_train)
        data_val, label_val = self._iterator_val.get_next()
        self._val_ops = trainer.createValidation(model, data_val, label_val)

        super().__init__(sess, model, batch_size, trainer, model_dir=model_dir,
                         save_checkpoint_interval=save_checkpoint_interval, log_interval=log_interval,
                         save_summary_interval=save_summary_interval)

        # To check the parameters of the model
        self.calcNumTrainableParams()

        # Get the models train summary ops.
        tf.summary.scalar('loss', self._training_ops[1])
        tf.summary.scalar('acc',  self._training_ops[2])
        self.__train_summary_op = tf.summary.merge_all()

    def doTraining(self, shot_size, shot_training_steps=1000, eval_interval=50, only_save_best_checkpoints=True):
        """
        Trains the model with the trainer and the input of the ModelSuit.
        :param shot_size: (Integer) The shot size of the classifier (number of samples per class to use in training).
        :param shot_training_steps: (Integer) Number of training steps for each shot size.
        :param eval_interval: (Integer) Every eval_interval steps the model will be evaluated.
        :param only_save_best_checkpoints: (Boolean) If true only the best model checkpoints on the evaluation set will
                                            be saved. True by default.
        """
        self._logger.train(
            "Started training for " + str(shot_training_steps) + " steps. Evaluation every " + str(eval_interval) +
            " steps...",
            "FsModelSuit:doTraining")

        start_training_time = time.time()
        start_log_loss_steps_time = time.time()

        # Get the small traing dataset.
        dataset_train = self._dataset_provider.getXShot("train", shot_size, self._sess, self._batch_size,
                                                        self._tfrecord_inputpipeline_config)

        data_train = np.reshape(np.array(dataset_train["data"]), [-1, np.prod(self.__shape_data_train)])

        label_train = np.reshape(np.array(dataset_train["label"]), [-1])
        onehot_labels = np.zeros((len(label_train), np.max(label_train) + 1), dtype=np.float32)
        onehot_labels[np.arange(len(label_train)), label_train] = 1.0
        label_train = np.array(onehot_labels, dtype=np.uint8)
        label_train = np.array(np.reshape(label_train, [-1, self.__shape_label_train[0]]))

        loss_value = 0
        acc_value = 0
        best_acc = 0
        try:
            # As long as the given train_steps are not reached.
            while self._global_step < shot_training_steps:
                # Run model training operations.
                p = np.random.RandomState(seed=42+self._global_step).permutation(len(label_train))
                data_train = data_train[p]
                label_train = label_train[p]

                _, loss_value, acc_value, summary = self._sess.run((self._training_ops[0], self._training_ops[1],
                                                                    self._training_ops[2], self.__train_summary_op),
                                                                   feed_dict={self._data_placeholder_train: data_train,
                                                                            self._label_placeholder_train: label_train})

                # If log_interval steps past print the logs.
                if self._global_step % self._log_interval == 0:
                    end_log_loss_steps_time = time.time()
                    self._logger.train("Step " + str(self._global_step) + ": " + str(self._log_interval) +
                                       " steps past in " + str(end_log_loss_steps_time-start_log_loss_steps_time) +
                                       "s. Accuracy: " + str(acc_value*100) + "%. Loss value: " + str(loss_value),
                                       "FsModelSuit:doTraining", self._global_step, shot_training_steps)
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

             # Save the model at the end. if only_save_best_checkpoints is not set.
            if not only_save_best_checkpoints:
                self._saver.save(self._sess, self._model_dir + '/checkpoints/model', global_step=self._global_step,
                                 write_meta_graph=False)

            # Stop training time.
            end_training_time = time.time()

            self._logger.train("Finished training for " + str(shot_training_steps) + " steps. Evaluation was every " +
                               str(eval_interval) + " steps. Training duration was: " +
                               str(end_training_time-start_training_time) + "s. Final accuracy: " + str(acc_value*100) +
                               "%. Final loss value: " + str(loss_value), "FsModelSuit:doTraining")

            self.doValidation("test")

        except tf.errors.OutOfRangeError:
            self._logger.warning(
                "Training iterator went out of range in step: " + str(self._global_step)+" befor the training finsihed!"
                , "FsModelSuit:doTraining")

    def doValidation(self, mode):
        """
        Validates the model on the subdataset subset defined by the mode.
        :param mode: (String) The subset of the dataset ("train", "eval" or "test).
        :return: acc: (Integer) The accuracy of the validation.
        """
        self._logger.val("Started validation for " + str(mode) + "...", "FsModelSuit:__doValidation")

        start_validation_time = time.time()
        start_log_loss_steps_time = time.time()

        val_step = 0
        acc_values = []
        loss_values = []
        try:
            # Init the validation iterator.
            self._sess.run(self._iterator_val.initializer,
                           feed_dict={self._file_placeholder_val: self._dataset_files[mode]})

            # Run validation until the iterator reaches the end of the subdataset.
            while True:
                # Run model validation operations.
                loss_value, acc_value = self._sess.run((self._val_ops[0], self._val_ops[1]))

                # If log_interval steps past print the logs.
                if (val_step % self._log_interval == 0) and (val_step != 0):
                    end_log_loss_steps_time = time.time()
                    self._logger.val("Step " + str(self._global_step) + ", " + str(mode) + " step " + str(val_step) +
                                     ": " + str(self._log_interval) + " " + str(mode) + " steps past in " +
                        str(end_log_loss_steps_time - start_log_loss_steps_time) + "s. Accuracy: " +
                        str(acc_value*100) + "%. Loss value: " + str(loss_value), "FsModelSuit:__doValidation")
                    start_log_loss_steps_time = time.time()

                # Append the validation results of the batch.
                loss_values.append(loss_value)
                acc_values.append(acc_value)

                # Increment the val_step.
                val_step += 1

        except tf.errors.OutOfRangeError:
            self._logger.val(
                "Validation complete. Iterator went out of range as wished in validation step: " + str(val_step),
                "FsModelSuit:__doValidation")

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
                str(end_validation_time-start_validation_time) + "s. Final accuracy: " +
                str(acc*100) + "%. Final loss value: " + str(loss),
                "FsModelSuit:__doValidation")

            # Log the validation results in a text file.
            self._summary_txt_writer.writeSummary("Acc for step " + str(val_step) + ": " + str(acc*100), mode)
            self._summary_txt_writer.writeSummary("Loss for step " + str(val_step) + ": " + str(loss), mode)
            return acc

    def infere(self, mode):
        """
        The model inferes on the given subdataset defined by the mode.
        :param mode: (String) The subset of the dataset ("train", "eval" or "test).
        :return: infers: (Array) The inference results. (subdataset_size, infers)
        :return: inputs: (Array) The inputs for the inference. (subdataset_size, inputs)
        :return: labels: (Array) The labels for the inference. (subdataset_size, labels)
        """
        # Reuse variables from the graph.
        with tf.variable_scope(tf.get_variable_scope()) as outer_scope:
            outer_scope.reuse_variables()

            start_log_loss_steps_time = time.time()

            inf_step = 1
            try:
                # Init the validation iterator.
                self._sess.run(self._iterator_val.initializer,
                               feed_dict={self._file_placeholder_val: self._dataset_files[mode]})

                # Define operations for inference.
                next_element_op, label_next_element_op = self._iterator_val.get_next()
                infere_op = self._model.getInferOp(next_element_op)

                # Run the inference once for the coming concat.
                infers, inputs, labels = self._sess.run((infere_op, next_element_op, label_next_element_op))

                # Run inference until the iterator reaches the end of the subdataset.
                while True:
                    # Run minference operations.
                    infer, input, label = self._sess.run((infere_op, next_element_op, label_next_element_op))

                    # If log_interval steps past print the logs.
                    if (inf_step % self._log_interval == 0):
                        end_log_loss_steps_time = time.time()
                        self._logger.val("Step " + str(self._global_step) + ", " + str(mode) + " step " + str(inf_step)
                                 + ": " + str(self._log_interval) + " " + str(mode) + " steps past in " +
                                str(end_log_loss_steps_time - start_log_loss_steps_time), "FsModelSuit:infere")
                        start_log_loss_steps_time = time.time()

                    # Inefficient concat of the batches
                    infers = np.concatenate((infers, infer))
                    inputs = np.concatenate((inputs, input))
                    labels = np.concatenate((labels, label))

                    # Increment the inf_step.
                    inf_step += 1

            except tf.errors.OutOfRangeError:
                self._logger.val(
                    "Inference complete. Iterator went out of range as wished in inference step: " + str(inf_step),
                    "FsModelSuit:infere")
                return infers, inputs, labels