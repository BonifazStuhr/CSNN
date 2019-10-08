import traceback

from LoggerNames import LoggerNames
from Logger_Component.SLoggerHandler import SLoggerHandler
from Experiment_Component.IExperiment import IExperiment
from ConfigInput_Component.ConfigProvider import ConfigProvider
from Experiment_Component.Models.Cifar10Reconstruction import Cifar10Reconstruction

from Experiment_Component.Experiments.csnnExperimentFunctions import prepareDataset, prepareEncoding,\
    getDatasetProvider, trainAndValReconstructionModel

class CsnnReconstructionExperiment(IExperiment):
    """
    The CsnnReconstructionExperiment test the given pre-trained CSNN by reconstructing the images from the encoding. The
    reconstructions are presented in Figure 6 of our paper: "CSNNs: Unsupervised, ackpropagation-Free Convolutional
    Neural Networks for Representation Learning".

    The experiment test each pre trained CSNN model for each given dataset by learning the defined reconstructor on the
    learned representation. If xFoldCrossValidation is set this will be repeated x times.

    :Attributes:
        __config:    (Dictionary) The config of the experiment, containing all model parameters. Refer to the config
                      csnnHypridExperiment.json as an example.
        __logger:    (Logger) The logger for the experiment.
        __num_gpus:  (Integer) The number of GPUs to use.
    """
    def __init__(self, config):
        """
        Constructor, initialize member variables.
        :param config: (Dictionary) The config of the experiment, containing all model parameters. Refer to the config
                        csnnReconstructionExperiment.json as an example.
        """
        self.__config = config
        self.__logger = SLoggerHandler().getLogger(LoggerNames.EXPERIMENT_C)
        self.__num_gpus = ConfigProvider().get_config("controllerConfig.json")["hardware"]["numGPUs"]

    def execute(self):
        """
        Executes the experiment with the given config.

        The experiment test each pre trained CSNN model for each given dataset by learning the defined reconstructor on
        the learned representation. If xFoldCrossValidation is set this will be repeated x times.
        """
        for csnn_config in self.__config["csnnConfigs"]:
            csnn_name = csnn_config["modelName"]

            try:
                for dataset_config in self.__config["datasetConfigs"]:
                    provider = getDatasetProvider(dataset_config)
                    if not dataset_config["nameOfDataset"] in csnn_config["batchSizes"].keys():
                        continue
                    for i in range(0, csnn_config["xFoldCrossValidation"]):
                        model_dir = "/" + csnn_name + "/" + dataset_config["nameOfDataset"] + "/xFoldCrossVal" + str(i)
                        self.__logger.info("Starting to test: " + model_dir, "CsnnReconstructionExperiment:execute")

                        if csnn_config["xFoldCrossValidation"] <= 1:
                            xseed = None
                        else:
                            xseed = 42 + i

                        dataset, dataset_generator = prepareDataset(provider, dataset_config, xfold_seed=xseed,
                                                                    augment_data=csnn_config["augmentData"])

                        dataset_max_div, _ = prepareDataset(provider, dataset_config, xfold_seed=xseed,
                                                            augment_data=csnn_config["augmentData"],
                                                            normalize_data="maxDiv")

                        self.__logger.info("Starting to create dataset encoding with: " + model_dir,
                                           "CsnnReconstructionExperiment:execute")
                        encoding_provider, encoding = prepareEncoding(csnn_config, dataset_generator, dataset,
                                                            dataset_config, csnn_name, self.__num_gpus,
                                                            model_dir + "/Csnn", zero_mean_unit_variance=
                                                            csnn_config["zeroMeanUnitVarianceEncoding"],
                                                            return_with_encoding=dataset_max_div)
                        self.__logger.info("Finished to create dataset encoding with: " + model_dir,
                                           "CsnnReconstructionExperiment:execute")

                        self.__logger.info("Starting to train reconstructor for: " + model_dir,
                                           "CsnnReconstructionExperiment:execute")
                        try:
                            rec_config = self.__config["reconstructionConfig"]
                            trainAndValReconstructionModel(Cifar10Reconstruction(rec_config), rec_config,
                                                           self.__num_gpus, model_dir + "/" + rec_config["modelName"] +
                                                        "/" + dataset_config["nameOfDataset"], encoding, dataset_config)

                        except Exception:
                            print(traceback.format_exc())
                        self.__logger.info("Finished to train reconstructor for: " + model_dir,
                                           "CsnnReconstructionExperiment:execute")
            except Exception:
                print(traceback.format_exc())


