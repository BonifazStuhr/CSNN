import os

class TxtSummaryWriter():
    """
    The TxtSummaryWriter manages the train, eval and test summaries written into simple text files.

     :Attributes:
        __summary_train_writer: (file_object) The summary writer for the training steps.
        __summary_eval_writer: (file_object) The summary writer for the eval steps.
        __summary_test_writer: (file_object) The summary writer for the test steps.
    """
    def __init__(self, modelDir):
        """
        Constructor, initialize member variables.
        :param modelDir: (String) The path to the model directory, where the summary will be saved under /logs.
        """

        if not os.path.exists(modelDir + "/txtlogs/"):
            os.makedirs(modelDir + "/txtlogs/")
        self.__summary_train_writer = open(modelDir + "/txtlogs/train.txt", "a")
        self.__summary_eval_writer = open(modelDir + "/txtlogs/eval.txt", "a")
        self.__summary_test_writer = open(modelDir + "/txtlogs/test.txt", "a")

    def writeSummary(self, string, mode):
        """
        Write the summary for the given mode.
        :param summary: (String) The string to write in the file.
        :param mode: (String) The mode for which the summary is saved.
        """
        string = string+"\n"
        if mode == "train":
            self.writeTrainSummary(string)
        elif mode == "eval":
            self.writeEvalSummary(string)
        elif mode == "test":
            self.writeTestSummary(string)

    def writeTrainSummary(self, string):
        """
        Write the summary for train mode.
        :param summary: (String) The string to write in the file.
        """
        self.__summary_train_writer.write(string)

    def writeEvalSummary(self, string):
        """
        Write the summary for eval mode.
        :param summary: (String) The string to write in the file.
        """
        self.__summary_eval_writer.write(string)

    def writeTestSummary(self, string):
        """
        Write the summary for test mode.
        :param summary: (String) The string to write in the file.
        """
        self.__summary_test_writer.write(string)

    def __exit__(self, *args):
        """
        To properly close the class when leaving 'with' statements.
        ::param *args
        """
        self.__summary_train_writer.close()
        self.__summary_eval_writer.close()
        self.__summary_test_writer.close()

