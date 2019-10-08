import tensorflow as tf

class TensorboardSummaryWriter():
    """
    The TensorboardSummaryWriter manages the train, eval and test summaries for the Tensorboard.

     :Attributes:
        __summary_train_writer: (tf.summary.FileWriter) The summary writer for the training steps.
        __summary_eval_writer: (tf.summary.FileWriter) The summary writer for the eval steps.
        __summary_test_writer: (tf.summary.FileWriter) The summary writer for the test steps.
    """
    def __init__(self, modelDir, graph):
        """
        Constructor, initialize member variables.
        :param modelDir: (String) The path to the model directory, where the summary will be saved under /logs.
        :param graph: (tf_graph) The graph of the model for which the summary is saved.
        """
        self.__summary_train_writer = tf.summary.FileWriter(modelDir + "/logs/train/", graph)
        self.__summary_eval_writer = tf.summary.FileWriter(modelDir + "/logs/eval/", graph)
        self.__summary_test_writer = tf.summary.FileWriter(modelDir + "/logs/test/", graph)

    def writeSummary(self, summary, global_step, mode, run_metadata=None):
        """
        Write the summary for the given mode.
        :param summary: (tf.Summary) The tf summary to write.
        :param global_step: (Integer) The current global_step.
        :param mode: (String) The mode for which the summary is saved.
        :param run_metadata: (tf.Metadata) Metadata of the run, e.g. execution times.
        """
        if mode == "train":
            self.writeTrainSummary(summary, global_step, run_metadata=run_metadata)
        elif mode == "eval":
            self.writeEvalSummary(summary, global_step, run_metadata=run_metadata)
        elif mode == "test":
            self.writeTestSummary(summary, global_step, run_metadata=run_metadata)

    def writeTrainSummary(self, summary, global_step, run_metadata=None):
        """
        Write the summary for train mode.
        :param summary: (tf.Summary) The tf summary to write.
        :param global_step: (Integer) The current global_step.
        :param run_metadata: (tf.Metadata) Metadata of the run, e.g. execution times.
        """
        self.__summary_train_writer.add_summary(summary, global_step)
        if run_metadata:
            self.__summary_train_writer.add_run_metadata(run_metadata, 'step%d' % global_step)

    def writeEvalSummary(self, summary, global_step, run_metadata=None):
        """
        Write the summary for eval mode.
        :param summary: (tf.Summary) The tf summary to write.
        :param global_step: (Integer) The current global_step.
        :param run_metadata: (tf.Metadata) Metadata of the run, e.g. execution times.
        """
        self.__summary_eval_writer.add_summary(summary, global_step)
        if run_metadata:
            self.__summary_eval_writer.add_run_metadata(run_metadata, 'step%d' % global_step)

    def writeTestSummary(self, summary, global_step, run_metadata=None):
        """
        Write the summary for test mode.
        :param summary: (tf.Summary) The tf summary to write.
        :param global_step: (Integer) The current global_step.
        :param run_metadata: (tf.Metadata) Metadata of the run, e.g. execution times.
        """
        if summary:
            self.__summary_test_writer.add_summary(summary, global_step)
        if run_metadata:
            self.__summary_test_writer.add_run_metadata(run_metadata, 'step%d' % global_step)