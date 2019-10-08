import traceback

from LoggerNames import LoggerNames
from Logger_Component.SLoggerHandler import SLoggerHandler
from Experiment_Component.IExperiment import IExperiment
from ConfigInput_Component.ConfigProvider import ConfigProvider

from Experiment_Component.Experiments.csnnExperimentFunctions import prepareDataset, getDatasetProvider, trainAndValHybrid

class CsnnHypridExperiment(IExperiment):
    """
    The CsnnHypridExperiment trains the cnn and hybrid model presented in Table 2 of our paper: "CSNNs: Unsupervised,
    Backpropagation-Free Convolutional Neural Networks for Representation Learning".

    The experiment trains each model for each given dataset like a CNN.
    If xFoldCrossValidation is set this will be repeated x times.

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
                        csnnHybridExperiment.json as an example.
        """
        self.__config = config
        self.__logger = SLoggerHandler().getLogger(LoggerNames.EXPERIMENT_C)
        self.__num_gpus = ConfigProvider().get_config("controllerConfig.json")["hardware"]["numGPUs"]

    def execute(self):
        """
        Executes the experiment with the given config.

        The experiment trains each model for each given dataset like a CNN. If xFoldCrossValidation is set this will be
        repeated x times.
        """
        for hybrid_config in self.__config["hybridConfigs"]:
            hybrid_name = hybrid_config["modelName"]

            try:
                for dataset_config in self.__config["datasetConfigs"]:
                    provider = getDatasetProvider(dataset_config)
                    if not dataset_config["nameOfDataset"] in hybrid_config["batchSizes"].keys():
                        continue
                    for i in range(0, hybrid_config["xFoldCrossValidation"]):
                        model_dir = "/" + hybrid_name + "/" + dataset_config["nameOfDataset"] + "/xFoldCrossVal" + str(i)
                        self.__logger.info("Starting to train: " + model_dir, "CsnnPerformancesExperiment:execute")

                        if hybrid_config["xFoldCrossValidation"] <= 1:
                            xseed = None
                        else:
                            xseed = 42 + i

                        dataset, dataset_generator = prepareDataset(provider, dataset_config, xfold_seed=xseed,
                                                                    augment_data=hybrid_config["augmentData"])

                        trainAndValHybrid(hybrid_config, dataset_generator, dataset, dataset_config, self.__num_gpus,
                                        model_dir+"/Csnn")
                        self.__logger.info("Finished to train: " + model_dir, "CsnnPerformancesExperiment:execute")

            except Exception:
                print(traceback.format_exc())




