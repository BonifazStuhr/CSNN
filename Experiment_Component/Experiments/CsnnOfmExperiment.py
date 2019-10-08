import traceback

from LoggerNames import LoggerNames
from Logger_Component.SLoggerHandler import SLoggerHandler
from Experiment_Component.IExperiment import IExperiment
from ConfigInput_Component.ConfigProvider import ConfigProvider
from Experiment_Component.Models.Mlp import Mlp

from Experiment_Component.Experiments.csnnExperimentFunctions import prepareDataset, trainAndValCsnn, prepareEncoding, \
    trainAndValClassifier, trainAndValFewShotClassifier, getDatasetProvider

class CsnnOfmExperiment(IExperiment):
    """
    The CsnnOfmExperiment trains the models used to create Figure 4 of our paper: "CSNNs: Unsupervised,
    Backpropagation-Free Convolutional Neural Networks for Representation Learning".

    The experiment trains each CSNN model for each given dataset for all defined training steps per layer and learns the
    defined classifier to test the learned representation. If xFoldCrossValidation is set this will be repeated x times.

    :Attributes:
        __config:    (Dictionary) The config of the experiment, containing all model parameters. Refer to the config
                      csnnOfmExperimentss.json as an example.
        __logger:    (Logger) The logger for the experiment.
        __num_gpus:  (Integer) The number of GPUs to use.

    """
    def __init__(self, config):
        """
        Constructor, initialize member variables.
        :param config: (Dictionary) The config of the experiment, containing all model parameters. Refer to the config
                        csnnOfmExperiment.json as an example.
        """
        self.__config = config
        self.__logger = SLoggerHandler().getLogger(LoggerNames.EXPERIMENT_C)
        self.__num_gpus = ConfigProvider().get_config("controllerConfig.json")["hardware"]["numGPUs"]

    def execute(self):
        """
        Executes the experiment with the given config.

        The experiment trains each CSNN model for each given dataset for all defined training steps per layer and learns
        thedefined classifier to test the learned representation. If xFoldCrossValidation is set this will be repeated
        x times.
        """
        encoding_config = self.__config["encodingConfig"]

        # For each model to train and val
        for csnn_config in self.__config["csnnConfigs"]:
            csnn_name = csnn_config["modelName"]
            try:
                # For each dataset
                for dataset_config in self.__config["datasetConfigs"]:

                    # Get the provider of the dataset
                    provider = getDatasetProvider(dataset_config)
                    if not dataset_config["nameOfDataset"] in csnn_config["batchSizes"].keys():
                        continue

                    # and for each xFold iteration
                    for i in range(0, csnn_config["xFoldCrossValidation"]):
                        # Construct the model Name
                        model_dir = "/" + csnn_name + "/" + dataset_config["nameOfDataset"] + "/xFoldCrossVal" + str(i)
                        self.__logger.info("Starting to train: " + model_dir, "CsnnOfmExperiment:execute")

                        # Reset the training of the model
                        csnn_config["trainingSteps"] = 0

                        # Train model for training_step_per_layer steps
                        prev_training_steps = 0
                        prev_training_steps_per_layer = 0
                        for training_step_per_layer in self.__config["ofmSteps"]:

                            # Based on the current training_step_per_layer we calculate the new training steps for
                            # non frozen layers
                            csnn_config["trainingSteps"] = training_step_per_layer*len(csnn_config["layers"])

                            # Calculate the training intervals based on the old training intervals.
                            for l in range(0, len(csnn_config["layers"])):
                                csnn_config["layers"][l]["trainInterval"][0] = prev_training_steps
                                prev_training_steps += (training_step_per_layer - prev_training_steps_per_layer)
                                csnn_config["layers"][l]["trainInterval"][1] = prev_training_steps

                            prev_training_steps_per_layer = training_step_per_layer
                            prev_training_steps = csnn_config["trainingSteps"]

                            # If xFoldCrossValidation is set, create a seed to shuffle the dataset.
                            if csnn_config["xFoldCrossValidation"] <= 1:
                                xseed = None
                            else:
                                xseed = 42+i
                            dataset, dataset_generator = prepareDataset(provider, dataset_config, xfold_seed=xseed,
                                                                        augment_data=csnn_config["augmentData"])

                            # Train the CSNN
                            trainAndValCsnn(csnn_config, dataset_generator, dataset, dataset_config, self.__num_gpus,
                                            model_dir+"/Csnn")
                            self.__logger.info("Finished to train: " + model_dir, "CsnnOfmExperiment:execute",
                                                "CsnnOfmExperiment:execute")

                            self.__logger.info("Starting to create dataset encoding with: " + model_dir,
                                               "CsnnOfmExperiment:execute")
                            # Create the encoding of the dataset via the trained CSNN.
                            encoding_provider = prepareEncoding(csnn_config, dataset_generator, dataset, dataset_config,
                                                                csnn_name, self.__num_gpus, model_dir+"/Csnn",
                                                                zero_mean_unit_variance=
                                                                csnn_config["zeroMeanUnitVarianceEncoding"])
                            self.__logger.info("Finished to create dataset encoding with: " + model_dir,
                                               "CsnnOfmExperiment:execute")

                            self.__logger.info("Starting to train classifiers for: " + model_dir,
                                               "CsnnOfmExperiment:execute")
                            # Test the trained model by training the classifier(s)
                            for classifier in self.__config["classifiers"]:
                                try:
                                    classifier["numClasses"] = dataset_config["numClasses"]
                                    if (classifier["type"] == "nonlinear") or (classifier["type"] == "linear"):
                                        trainAndValClassifier(Mlp(classifier),
                                                              classifier, encoding_provider, encoding_config,
                                                              self.__num_gpus, model_dir+"/"+classifier["modelName"]
                                                              +str(training_step_per_layer))
                                    elif classifier["type"] == "fewShot":
                                        trainAndValFewShotClassifier(Mlp(classifier),
                                                                     classifier, encoding_provider, encoding_config,
                                                                     self.__num_gpus, model_dir+"/"
                                                                     +classifier["modelName"]+str(training_step_per_layer))
                                except:
                                    print(traceback.format_exc())
                            self.__logger.info("Finished to train classifiers for: " + model_dir,
                                               "CsnnOfmExperiment:execute")
            except:
                print(traceback.format_exc())




