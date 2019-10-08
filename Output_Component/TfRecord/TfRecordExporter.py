import tensorflow as tf
import numpy as np

from LoggerNames import LoggerNames
from Logger_Component.SLoggerHandler import SLoggerHandler

class TfRecordExporter:
    """
    The TfRecordExporter exports given datasets into a tfrecord files.

    :Attributes:
        __logger:                 (Logger) The logger for the controller.
        __num_entries:            (Integer) The number of entries in the dataset to export.
        __num_written_entries:    (Integer) The number of already written data_entries.
        __tf_record_writer:       (tf.python_io.TFRecordWriter) The writer for the entries.
    """

    def __init__(self, tfrecord_dir, tfrecord_name, datasset_split, mode):
        """
        Constructor, initialize member variables.
        :param tfrecord_dir: (String) The path to the record save directory.
        :param tfrecord_name: (String) The name of the saved record.
        :param datasset_split: (Array) The splits of the datasets in format [trainNum,evalNum,testNum].
        :param mode: (String) The mode of the saved record.
        :param num_entries: (Integer) The number of entries in the dataset to export.
        """
        self.__logger = SLoggerHandler().getLogger(LoggerNames.Output_C)

        # Variables to see the progress.
        self.__num_written_entries = 0

        # Save path of the records.
        tfrecord_savepath = str(tfrecord_dir) + "/" + str(tfrecord_name) + "/" + str(tfrecord_name) + "_" + \
                            str(datasset_split) + "_" + mode

        # Create record writer for the dataset.
        tf_record_writer_options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)
        self.__tf_record_writer = tf.python_io.TFRecordWriter(tfrecord_savepath + ".tfrecords",
                                                              tf_record_writer_options)

    def writeEntry(self, entry):
        """
        Writes a given entry to the a tfrecord file specified in the writer.
        :param entry: (Dictionary) The entry to write, e.g.: {"data": [1,2,3,4], "label": 1}
        """
        # Constructing the entry.
        feature = {}
        for entry_part_name, entry_part_value in entry.items():
            feature[entry_part_name] = tf.train.Feature(bytes_list=tf.train.BytesList(
                value=[entry_part_value.tostring()]))
            feature[entry_part_name+"Shape"] = tf.train.Feature(int64_list=tf.train.Int64List(
                value=entry_part_value.shape))

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        self.__tf_record_writer.write(example.SerializeToString())

    def writeData(self, data, shuffle=False):
        """
        Writes the given Data to the a tfrecord file specified in the writer.
        :param entry: (Dictionary) The entry to write, e.g.: {"data": [[1,2,3,4],[1,2,9,4],] "label": [1,2]}
        """
        entry_part_names = list(data.keys())
        num_entry_parts_per_entry = len(entry_part_names)

        if shuffle:
            # Randomize dataset order.
            p = np.random.RandomState(seed=42).permutation(len(data[entry_part_names[0]]))
            for entry_part_name in entry_part_names:
                data[entry_part_name] = data[entry_part_name][p]

        entry_part_values = list(data.values())

        # Writing the entry for each sample in the dataset.
        # convert from {"data": [[1,2,3,4],[1,2,9,4],] "label": [1,2]} into {"data": [1,2,3,4], "label": [1]},
        # {"data": [1,2,9,4], "label": [2]}
        for i in range(len(entry_part_values[0])):
            entry = {}
            for j in range(num_entry_parts_per_entry):
                entry[entry_part_names[j]] = entry_part_values[j][i]
            self.writeEntry(entry)

    def close(self):
        """
        Closes the TfRecordsExporter and the tf.python_io.TFRecordWriter properly.
        """
        self.__tf_record_writer.close()
        self.__tf_record_data_writer = None

    def __enter__(self):
        """
        To properly use the class in 'with' statements.
        :returns self: (TfRecordExporter) The class itself.
        """
        return self

    def __exit__(self, *args):
        """
        To properly close the class when leaving 'with' statements.
        ::param *args
        """
        self.close()

