import traceback
import os
import sys

import numpy as np

from LoggerNames import LoggerNames
from Logger_Component.SLoggerHandler import SLoggerHandler
from Experiment_Component.IExperiment import IExperiment
from ConfigInput_Component.ConfigProvider import ConfigProvider
from Experiment_Component.Experiments.csnnExperimentFunctions import prepareDataset, prepareEncoding,\
    getDatasetProvider

class CsnnVisualizationExperiment(IExperiment):
    """
    The CsnnVisualizationExperiment creates mean and sub neuron activities for each label for the given pre-trained
    CSNNs. These activities are presented in Figure 6 of our paper: "CSNNs: Unsupervised, Backpropagation-Free
    Convolutional Neural Networks for Representation Learning".

    The experiment creates mean and sub neuron activities for each label for each given dataset. If xFoldCrossValidation
    is set this will be repeated x times.

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
                        csnnVisualizationsExperiment.json as an example.
        """
        self.__config = config
        self.__logger = SLoggerHandler().getLogger(LoggerNames.EXPERIMENT_C)
        self.__num_gpus = ConfigProvider().get_config("controllerConfig.json")["hardware"]["numGPUs"]

    def execute(self):
        """
        Executes the experiment with the given config.

        The experiment t creates mean and sub neuron activities for each label for each given dataset.
        If xFoldCrossValidation is set this will be repeated x times.
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
                        self.__logger.info("Starting to train: " + model_dir, "CsnnVisualizationExperiment:execute")

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
                                           "CsnnVisualizationExperiment:execute")
                        encoding_provider, encoding = prepareEncoding(csnn_config, dataset_generator, dataset,
                                                                    dataset_config, csnn_name, self.__num_gpus,
                                                                    model_dir + "/Csnn", zero_mean_unit_variance=
                                                                    csnn_config["zeroMeanUnitVarianceEncoding"],
                                                                    return_with_encoding=dataset_max_div)

                        self.__logger.info("Finished to create dataset encoding with: " + model_dir,
                                           "CsnnVisualizationExperiment:execute")

                        self.__logger.info("Starting to create mean activities for: " + model_dir,
                                           "CsnnVisualizationExperiment:execute")
                        enc_for_label = {}
                        for i in range(0, len(encoding["y_test"])):
                            enc_for_label.setdefault(np.argmax(encoding["y_test"][i]), []).append(encoding["x_test"][i])

                        mean_ecs_for_label = []
                        for key, value in sorted(enc_for_label.items()):
                            mean_ec_for_label = np.mean(np.mean(np.mean(value, axis=0), axis=0), axis=0)
                            grid = csnn_config["layers"][-1]["somGrid"]
                            mean_ec_for_label = np.reshape(mean_ec_for_label, [grid[0], grid[1], grid[2]])
                            mean_ecs_for_label.append(mean_ec_for_label)

                        # Dir to save and reload model.
                        save_path = os.path.dirname(
                            sys.modules['__main__'].__file__) + "/experimentResults" + model_dir
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        np.save(save_path + "/test_mean_acts", np.array(mean_ecs_for_label))
                        self.__logger.info("Finised to create mean activities for: " + model_dir,
                                           "CsnnVisualizationExperiment:execute")

                        self.__logger.info("Starting to create mean sub activities for: " + model_dir,
                                           "CsnnVisualizationExperiment:execute")
                        sub_encs = []
                        for enc in mean_ecs_for_label:
                            sub_encs_for_label = []
                            for kenc2 in mean_ecs_for_label:
                                sub_enc = enc - kenc2
                                sub_encs_for_label.append(sub_enc)
                            sub_encs.append(sub_encs_for_label)

                        np.save(save_path + "/test_sub_mean_acts", np.array(sub_encs))
                        self.__logger.info("Finished to create mean sub activities for: " + model_dir,
                                           "CsnnVisualizationExperiment:execute")

            except Exception:
                print(traceback.format_exc())




