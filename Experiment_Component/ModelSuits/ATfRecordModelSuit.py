import tensorflow as tf

from Experiment_Component.AModelSuit import AModelSuit

class ATfRecordModelSuit(AModelSuit):
    """
    A ATfRecordModelSuit handles the train/eval/test/inference of a model trained with a Tensorflow pipeline on
    tfrecord files. Therefore it brings the DatasetProvider, the model and the trainer, together in one place.

    :Attributes:
        _dataset_provider:                 (ADataProvider) The provider of the tfrecord input.
        _tfrecord_inputpipeline_config:    (Dictionary) The configuration of the input pipeline.
        _dataset_files:                    (Array) The names of the tf record files for the file_placehoders.
        _iterator_train:                   (tf.data.Iterator) The iterator for the training.
        _file_placeholder_train:           (tf.placeholder) The placeholder for the dataset_files from iterator_train.
        _iterator_val:                     (tf.data.Iterator) The iterator for the validation.
        _file_placeholder_val:             (tf.placeholder)The placeholder for the dataset_files from iterator_val.
        _training_ops:                     (Tensor) The training operation fpr the model created from the ITrainer.
        _val_ops:                          (Tensor) The validation operation fpr the model created from the ITrainer.
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
        self._iterator_train, self._file_placeholder_train  = \
            dataset_provider.getTfRecordInputPipelineIteratorsAndPlaceholdersFor("train", batch_size,
                                                                          tfrecord_inputpipeline_config["train"])
        self._iterator_val, self._file_placeholder_val,  = \
            dataset_provider.getTfRecordInputPipelineIteratorsAndPlaceholdersFor("val", batch_size,
                                                                          tfrecord_inputpipeline_config["val"])

        # Build training_ops via trainer.
        data_train, label_train = self._iterator_train.get_next()
        self._training_ops = trainer.createTraining(model, data_train, label_train)
        data_val, label_val = self._iterator_val.get_next()
        self._val_ops = trainer.createValidation(model, data_val, label_val)

        super().__init__(sess, model, batch_size, trainer, model_dir=model_dir,
                         save_checkpoint_interval=save_checkpoint_interval, log_interval=log_interval,
                         save_summary_interval=save_summary_interval)

        # To check the parameters of the model
        self.calcNumTrainableParams()
