import traceback

from LoggerNames import LoggerNames
from Logger_Component.SLoggerHandler import SLoggerHandler
from ConfigInput_Component.ConfigProvider import ConfigProvider
from Experiment_Component.ExperimentScheduler import ExperimentScheduler
from Output_Component.TfRecord.TfRecordHandler import TfRecordHandler

class Controller:
    """
    The controller is the central point of the framework. It takes care of the execution of various programs like
    the creation of TfRecord datasets or the execution of experiments via a scheduler.

    :Attributes:
        __controller_config_file_path: (String) The path to the config for the controller.
        __config:                      (Dictionary) The config of the controller.
        __logger:                      (Logger) The logger for the controller.
        __experiment_scheduler:        (ExperimentScheduler) The scheduler to handle multiple experiments.
        __tf_record_handler:           (TfRecordHandler) The handler to create the tf records.
        __config_provider:             (ConfigProvider) : The provider to request config input.
        __experiment_scheduler:        (ExperimentScheduler) The scheduler to schedule the experiment execution.
    """
    def __init__(self, controller_config_file_path):
        """
        Constructor, initialize member variables.
        :param controller_config_file_path: (String) String to controllerConfig File.
        """
        print("Controller: Starting __init__() ...")
        self.__controller_config_file_path = controller_config_file_path
        self.__config = None
        self.__logger = None
        self.__tf_record_handler = None
        self.__config_provider = None
        self.__experiment_scheduler = None
        print("Controller: Finished __init__()")

    def init(self):
        """
        Init method, initialize member variables and other program parts.
        :return: successful: (Boolean) Was the execution successful?
        """
        print("Controller: Starting init() ...")
        self.__logger = SLoggerHandler().getLogger(LoggerNames.CONTROLLER_C)
        self.__logger.info("Loading config ...", "Controller:init")
        successful = True

        try:
            self.__config_provider = ConfigProvider()
            self.__config = self.__config_provider.get_config(self.__controller_config_file_path)
            self.__logger.info("Finished loading config.", "Controller:init")

            self.__tf_record_handler = TfRecordHandler(tfrecord_dir="data", dataset_prepreprocessors=self.__config[
                "datasetPrePreProcessors"], num_threads=self.__config["hardware"]["numCPUCores"])

            self.__logger.info("Finished init()", "Controller:init")
        except:
            successful = False
            self.__logger.error("Canceled init(). An error accrued!", "Controller:init")
            print(traceback.format_exc())

        return successful

    def execute(self):
        """
        Executes the execution specified in the controllers config.
        :return: successful: (Boolean) Was the execution successful??
        """
        self.__logger.info("Starting execute() ...", "Controller:execute")
        successful = True
        if self.__config["executeCreateTfRecordsFromDataset"]:
            try:
                self.__logger.info("Starting executeCreateTfRecordsFromDataset() ...", "Controller:execute")
                self.__tf_record_handler.createTfRecords(self.__config["datasetsToCreateTfRecords"],
                                                         self.__config["datasetTfRecordSplits"])
                self.__logger.info("Finished executeCreateTfRecordsFromDataset()", "Controller:execute")
            except:
                successful = False
                self.__logger.error("Canceled executeCreateTfRecordsFromDataset(). An error accrued!",
                                    "Controller:execute")
                print(traceback.format_exc())

        if self.__config["executeExperiments"]:
            try:
                self.__logger.info("Starting executeExperiments() ...", "Controller:execute")
                # load schedule
                experiment_schedule = self.__config_provider.get_config("experimentSchedule.json")
                self.__experiment_scheduler = ExperimentScheduler(experiment_schedule)
                self.__experiment_scheduler.execute()
                self.__logger.info("Finished executeExperiments()", "Controller:execute")
            except:
                successful = False
                self.__logger.error("Canceled executeExperiments(). An error accrued!", "Controller:execute")
                print(traceback.format_exc())

        return successful
