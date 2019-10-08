import os
import sys
import glob
import shutil
import tensorflow as tf
from abc import ABCMeta, abstractmethod

from LoggerNames import LoggerNames
from Logger_Component.SLoggerHandler import SLoggerHandler
from Output_Component.Summary.TensorboardSummaryWriter import TensorboardSummaryWriter
from Output_Component.Summary.TxtSummaryWriter import TxtSummaryWriter
from Output_Component.Summary.TxtFunctionTimeStopper import TxtFunctionTimeStopper

class AModelSuit(metaclass=ABCMeta):
    """
    A AModelSuit handles the train/eval/test/inference of a model. Therefore it brings the input, the model and
    the trainer, together in one place. In each AModelSuit functions for the training and validation must be defined.

    The AModelSuit provides basic functionality like session handling and model saving and defines
    interface methods for ModelSuits.

    :Attributes:
        _model:                        ("Model") The model to handle with the ModelSuit
        _logger:                       (Logger) The logger for the ModelSuit.
        _sess:                         (tf.Session) The Tensorflow session for the execution of the graph constructed
                                        by the ModelSuit
        _trainer:                      (ITrainer) The trainer to train the model.
        _batch_size:                   (Integer) The batch size for the model.
        _modelDir:                     (String) The directory of the model (e.g. to save it).
        _log_interval:                 (Integer) Every log_interval steps the ModelSuit writes logs.
        _save_summary_interval:        (Integer) Every save_summary_interval steps the ModelSuit saves
                                       Tensorboard summaries.
        _save_checkpoint_interval:     (Integer) Every _save_checkpoint_interval steps the ModelSuit saves model
                                       (training) checkpoints.
        _saver:                        (tf.train.Saver) The saver to save the model.
        _globalStep:                   (Integer) The current global step of the model.
        _summary_writer:               (TensorboardSummaryWriter) The writer for the Tensorboard summaries.
        _summary_txt_writer:           (TxtSummaryWriter) The writer for the text summaries.
        _txt_function_time_stopper:    (TxtFunctionTimeStopper) The writer and stopper of function times.
    """

    def __init__(self, sess, model, batch_size, trainer, model_dir="/model", save_checkpoint_interval=500,
                 log_interval=100, save_summary_interval=250):
        """
        Constructor, initialize member variables.
        :param sess:  (tf.Session) The Tensorflow session for the execution of the graph constructed by the ModelSuit
        :param model: ("Model") The model to handle with the ModelSuit
        :param batch_size: (Integer) The batch size for the model.
        :param trainer: (ITrainer) The trainer to train the model.
        :param model_dir: (String) The directory of the model (e.g. to save it). "/model" by default.
        :param save_checkpoint_interval: (Integer) Every _save_checkpoint_interval steps the ModelSuit saves model
                                        (training) checkpoints. 500 by default.
        :param log_interval: (Integer) Every log_interval steps the ModelSuit writes logs. 100 by default.
        :param save_summary_interval: (Integer) Every save_summary_interval steps the ModelSuit saves Tensorboard
                                       summaries. 250 by default.
        """
        # Set model and session.
        self._sess = sess
        self._model = model

        # Setting up the Loggers
        self._logger = SLoggerHandler().getLogger(LoggerNames.EXPERIMENT_C)

        # Hooks for train, eval and predict
        self._trainer = trainer
        self._batch_size = batch_size

        # Dir to save and reload model.
        self._model_dir = os.path.dirname(sys.modules['__main__'].__file__) + "/experimentResults" + model_dir

        # Log every log_interval steps
        self._log_interval = log_interval

        # Log summary every save_summary_interval steps
        self._save_summary_interval = save_summary_interval

        # To save model and checkpoints.
        self._save_checkpoint_interval = save_checkpoint_interval
        self._saver = tf.train.Saver(max_to_keep=10)
        self._global_step = 0

        # Restore existing Model if its not half or wrong defined.
        if len(glob.glob(self._model_dir + "/checkpoints/model-*")) >= 1:
            # Restore model weights from previously saved model
            self._saver.restore(self._sess, tf.train.latest_checkpoint(self._model_dir + '/checkpoints/'))

            checkpoint = tf.train.get_checkpoint_state(self._model_dir + '/checkpoints/')

            # Extract from checkpoint filename.
            self._global_step = int(os.path.basename(checkpoint.model_checkpoint_path).split('-')[1])

        # Create new Model.
        else:
            # If there is some half or wrong defined model stuff remove it.
            if os.path.exists(self._model_dir):
                shutil.rmtree(self._model_dir)
            os.makedirs(self._model_dir + '/checkpoints/')
            self._sess.run(tf.global_variables_initializer())
            self._saver.save(self._sess, self._model_dir + '/checkpoints/model')

        # To save summary.
        self._summary_writer = TensorboardSummaryWriter(self._model_dir, self._sess.graph)
        self._summary_txt_writer = TxtSummaryWriter(self._model_dir)
        self._txt_function_time_stopper = TxtFunctionTimeStopper(self._model_dir)

    @abstractmethod
    def doTraining(self, train_steps, eval_interval, only_save_best_checkpoints):
        """
        Interface Method: Trains the model with the trainer and the input of the ModelSuit.
        :param train_steps: (Integer) The steps to train the model.
        :param eval_interval: (Integer) Every eval_interval steps the Model will be evaluated.
        :param only_save_best_checkpoints: (Boolean) If true only the best Model checkpoints on the evaluation set will
                                            be saved.
        """
        pass

    @abstractmethod
    def doValidation(self, mode):
        """
        Interface Method: Validates the model on the subdataset subset defined by the mode.
        :param mode: (String) The subset of the dataset ("train", "eval" or "test).
        """
        pass

    def doDatesetValidation(self):
        """
        Validates the model on the entire dataset.
        """
        self.doValidation("train")
        self.doValidation("eval")
        self.doValidation("test")

    def closeSession(self):
        """
        Closes the Tensorflow session.
        """
        self._sess.close()
        tf.reset_default_graph()

    def calcNumTrainableParams(self):
        """
        Calculates and logs the number of trainable parameters in the model.
        """
        total_parameters = 0
        self._logger.debug("Calculating trainable parameters ...", "AModelSuit:calcNumTrainableParams")
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            self._logger.debug(
                "For Variable: " + str(variable) + " with Shape: " + str(shape) + " with length: " + str(len(shape)),
                "AModelSuit:calcNumTrainableParams")
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
                self._logger.debug("For Dimension: " + str(dim) + " found: " + str(
                    dim.value) + " parameters. Total for shape so far: " + str(variable_parameters),
                                   "AModelSuit:calcNumTrainableParams")
                total_parameters += variable_parameters
            self._logger.debug("Total number of trainable parameters in model so far: " + str(total_parameters),
                               "AModelSuit:calcNumTrainableParams")
        self._logger.debug("Total number of trainable parameters in model: " + str(total_parameters),
                           "AModelSuit:calcNumTrainableParams")
