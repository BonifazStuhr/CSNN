import importlib
import math
import numpy as np

from utils.MultiThreading.FunctionThread import FunctionThread
from LoggerNames import LoggerNames
from Logger_Component.SLoggerHandler import SLoggerHandler
from Output_Component.TfRecord.TfRecordExporter import TfRecordExporter

class TfRecordHandler:
    """
    The TfRecordHandler handles the creation of tfrecords for various datasets.

    :Attributes:
        __logger:       (Logger) The logger for the controller.
        __tfrecord_dir: (String) The path to the record save directory.
        __num_threads:  (Integer) The number of threads for writing the tfrecords. 1 by default.
    """
    def __init__(self, tfrecord_dir, dataset_prepreprocessors=None, num_threads=0):
        """
        Constructor, initialize member variables.
        :param tfrecord_dir: (String) The path to the record save directory.
        :param dataset_prepreprocessors: (Dictionary) The prepreprocessors to use in the corresponding splits.
                                          None by default.
        :param num_threads: (Integer) The number of threads for writing the tfrecords. 1 by default.
        """
        self.__logger = SLoggerHandler().getLogger(LoggerNames.Output_C)
        self.__tfrecord_dir = tfrecord_dir
        self.__dataset_prepreprocessors = dataset_prepreprocessors
        self.__num_threads = num_threads

    def createTfRecords(self, dataset_names, dataset_splits):
        """
        Creates the tfrecords for the given datasets witch the given train, eval, test split.
        Therefore DataProviders with the names "dataset_nameProvider" for example "MnistProvider" must be implemented
        properly for the datasets to convert them into tfrecords.
        :param dataset_names: (String) The name of the dataset
        :param dataset_splits: (Dictionary) The splits of the datasets in format
                               {"datasetname":[trainNum,evalNum,testNum, "prepreprocessing"].
        """
        # Calcualte to number of splits to write in tfrecords.
        splits_to_write = 0
        for dataset_name in dataset_names:
            splits_to_write += len(dataset_splits[dataset_name])
            self.__logger.info("Read " + str(len(dataset_splits[dataset_name])) + " splits to write for dataset " +
                               str(dataset_name) + ": " +str(dataset_splits[dataset_name]),
                               "TfRecordHandler:createTfRecords")
        self.__logger.info("Read " + str(splits_to_write) + " splits to write in total.",
                           "TfRecordHandler:createTfRecords")

        # Calculate the threads_per_split roughly.
        threads_per_split = self.__num_threads / splits_to_write
        self.__logger.info("Splits to write: "+str(splits_to_write)+ ". Threads available: " + str(self.__num_threads) +
                           ". => Calculated  " + str(threads_per_split) + " threads per split.",
                           "TfRecordHandler:createTfRecords")

        # If there is just one split to write, there is no need to multithread multible dataset splits, just the split
        # itself.
        if splits_to_write == 1:
            self.__logger.info("Just one dataset split to write. Starting to create tfrecords with multithreading in "
                               "dataset parts...", "TfRecordHandler:createTfRecords")
            for dataset_name in dataset_names:
                for dataset_split in dataset_splits[dataset_name]:
                    self.createTfRecord(dataset_name, dataset_split, threads_per_split)

        # If number of threads is one or less then one, there is no need to multithead anything.
        elif self.__num_threads <= 1:
            self.__logger.info("Less then or 1 thread available in total. Starting to create tfrecords without "
                               "multithreading...", "TfRecordHandler:createTfRecords")
            for dataset_name in dataset_names:
                for dataset_split in dataset_splits[dataset_name]:
                    self.createTfRecord(dataset_name, dataset_split, 0)

        # If the number of threads is greater then te split to write and there a so many thread, that each split can
        # have more then 3 threads, then execute each split in a extra thread and distribute the training, eval and test data
        # file writing in the thread into 2-3 threads.
        elif threads_per_split >= 3:
            self.__logger.info("More then 3 threads per split! Starting to create tfrecords with " +
                               str(self.__num_threads) + " threads. Multithreading in splits and dataset parts...",
                               "TfRecordHandler:createTfRecords")
            # Calculate threads per split "exactly".
            threads_per_split = max(math.ceil(threads_per_split - 1), 3)

            # Run each split in a thread with threads_per_split threads per split.
            threads = []
            for dataset_name in dataset_names:
                for dataset_split in dataset_splits[dataset_name]:
                    threads.append(FunctionThread(self.createTfRecord, dataset_name, dataset_split, threads_per_split))
                    threads[-1].start()

            # Wait for all threads to finish.
            for thread in threads:
                thread.join()

        # If the number of threads is more the 2, then execute as many splits as possible in parallel, if some threads
        # are idle because the number of threads in a little bit greater then the number of splits, distribute the
        # threads in two-pairs to the writing of the training, eval and test files.
        else:
            self.__logger.info("More then 1 thread available in total! Started to create tfrecords with " +
                               str(self.__num_threads) + " threads. Multithreading in splits and sometimes in dataset...",
                               "TfRecordHandler:createTfRecords")
            # Calculate the threads per split "exactly".
            threads_for_splits = self.__calculateThreadsForEachSplit(splits_to_write)

            # Prepare the queue.
            queue = []
            index = 0
            for dataset_name in dataset_names:
                for dataset_split in dataset_splits[dataset_name]:
                    queue.append([self.createTfRecord, dataset_name, dataset_split, threads_for_splits[index]])
                    index += 1

            # Run the splits in as many threads as possible.
            threads = []
            fifo_index = 1
            for i in range(1, len(queue) + 1):
                threads.append(FunctionThread(queue[i - 1][0], queue[i - 1][1], queue[i - 1][2], queue[i - 1][3]))
                threads[-1].start()

                self.__logger.info("Started split thread " + str(i) + "...", "TfRecordHandler:createTfRecord")
                # Wait for the latest started thread to finish, this makes sense because the train data thread is
                # started first and the the data to process in the training set is usually the most.
                if i >= self.__num_threads:
                    # Just wait for the thread to finish and print the info if its not the last thread.
                    if i < len(queue):
                        self.__logger.info("Waiting for split thread " + str(i) + " to finish to start next thread...",
                                            "TfRecordHandler:createTfRecord")
                        threads[fifo_index].join()
                        fifo_index += 1

            # Wait for all threads to finsih.
            self.__logger.info("Waiting for all remaining split threads to finish...", "TfRecordHandler:createTfRecord")
            for thread in threads:
                thread.join()

        self.__logger.info("Finished to create all tfrecords.", "TfRecordHandler:createTfRecords")

    def createTfRecord(self, dataset_name, dataset_split, num_threads=1):
        """
        Creates the tfrecords for the given dataset witch the given split.
        Therefore a DataProvider with the name "datasetnameProvider" for example MnistProvider must be implemented
        properly for the dataset to convert it into tfrecords.
        :param dataset_names : (String) The name of the dataset
        :param dataset_split : (Array) The splits of the datasets in format [trainNum,evalNum,testNum,
                              "prepreprocessing"].
        :param num_threads : (Integer) The number of threads for writing the tfrecords. 1 by default.
        """
        self.__logger.info("Started to create tfrecords for " + str(dataset_name) + " with split  " + str(
            dataset_split) + " and " + str(num_threads) + " extra threads per split...",
                           "TfRecordHandler:createTfRecords")

        # The train, test and eval part of the dataset is read into three separate files, therefore it makes no sense to
        # use more then three threads.
        if num_threads > 3:
            num_threads = 3

        # The name of the provider class.
        provider_name = dataset_name+"Provider"

        # Dynamically import the provider class by name.
        provider_module = importlib.import_module("Input_Component.DataProviders."+provider_name)

        # Dynamically load the provider class by name.
        # Combined with the above import its like: from Input_Component.DataProviders.MnistProvider import MnistProvider.
        dataset_provider = getattr(provider_module, provider_name)()

        # Set the Split of the Dataset.
        dataset_provider.setDatasetSplit(dataset_split)

        if dataset_split[3] != "None":
            # Dynamically import the preprocessing class by name.
            preprocessing_name = self.__dataset_prepreprocessors[dataset_split[3]]["preProcessingClassName"]
            preprocessing_module = importlib.import_module(
            "Preprocessing_Component.Prepreprocessing." + preprocessing_name)

            # Dynamically load the class by name.
            prepreprocessor = getattr(preprocessing_module, preprocessing_name)\
                (self.__dataset_prepreprocessors[dataset_split[3]])

        # Prepare to Write split in tfrecord files.
        queue = []
        if dataset_provider.datasetProcessableAtOnce():
            # Load test data.
            dataset = dataset_provider.getSplittedDatasetInNumpy()
            train_data = {"data": dataset["x_train"], "label": dataset["y_train"]}
            eval_data = {"data": dataset["x_eval"], "label": dataset["y_eval"]}
            test_data = {"data": dataset["x_test"], "label": dataset["y_test"]}

            if dataset_split[3] != "None":
                #Todo: Multithreading
                train_data, eval_data, test_data = prepreprocessor.process(train_data, eval_data, test_data)
                dataset_split[0] = len(train_data["data"])
                dataset_split[1] = len(eval_data["data"])
                dataset_split[2] = len(test_data["data"])

            # Debug_outdated
            #from Logger_Component.DataVisualizers.ImagePlusLabelVisualizer import ImagePlusLabelVisualizer
            #ImagePlusLabelVisualizer().visualizeImagesAndLabelsWithBreak(train_data, train_labels, [0, 7, 3])
            #ImagePlusLabelVisualizer().visualizeImagesAndLabelsWithBreak(eval_data, eval_labels, [0, 7, 3])
            #ImagePlusLabelVisualizer().visualizeImagesAndLabelsWithBreak(test_data, test_labels, [0, 7, 3])

            # If threading is on prepare the queue.
            if num_threads > 0:
                queue.append([self.__writeTfrecord, train_data, "train", dataset_name, dataset_split])
                queue.append([self.__writeTfrecord, eval_data, "eval", dataset_name, dataset_split])
                queue.append([self.__writeTfrecord, test_data, "test", dataset_name, dataset_split])
            else:
                self.__writeTfrecord(train_data, "train", dataset_name, dataset_split)
                self.__writeTfrecord(eval_data, "eval", dataset_name, dataset_split)
                self.__writeTfrecord(test_data, "test", dataset_name, dataset_split)
        else:

            #Todo add preprocessing

            # If threading is on prepare the queue.
            if num_threads > 0:
                queue.append([self.__writeTfrecordBatchwise, dataset_provider, "train", dataset_name, dataset_split])
                queue.append([self.__writeTfrecordBatchwise, dataset_provider, "eval", dataset_name, dataset_split])
                queue.append([self.__writeTfrecordBatchwise, dataset_provider, "test", dataset_name, dataset_split])
            else:
                self.__writeTfrecordBatchwise(dataset_provider, "train", dataset_name, dataset_split)
                self.__writeTfrecordBatchwise(dataset_provider, "eval", dataset_name, dataset_split)
                self.__writeTfrecordBatchwise(dataset_provider, "test", dataset_name, dataset_split)

        # If threading is on, start the maximum amount of threads and wait for all threads to finish.
        threads = []
        if num_threads > 0:
            for i in range(1, len(queue)+1):
                threads.append(FunctionThread(queue[i-1][0], queue[i-1][1],  queue[i-1][2],  queue[i-1][3],
                                              queue[i-1][4]))
                threads[-1].start()

                self.__logger.info("Started dataset thread " + str(i) + " for " + str(dataset_name) + " with split  "
                                   + str(dataset_split) + "...", "TfRecordHandler:createTfRecord")

                # Wait for the latest started thread to finish, this makes sense because the train data thread is
                # started first and the the data to process in the training set is usually the most.
                if i >= num_threads:
                    # Just wait for the thread to finish and print the info if its not the last thread.
                    if i < 3:
                        self.__logger.info("Waiting for dataset thread " + str(i) +
                                           " to finish to start next thread for " + str(dataset_name) + " with split "
                                           + str(dataset_split)+"...", "TfRecordHandler:createTfRecord")
                        threads[i-1].join()

            self.__logger.info("Waiting for all remaining dataset threads to finish for " +
                                str(dataset_name) + " with split " + str(dataset_split)+"...",
                               "TfRecordHandler:createTfRecord")

            # Wait for all threads to finsih.
            for thread in threads:
                thread.join()

        self.__logger.info("Finished to create tfrecords for " + str(dataset_name) + " with split  " + str(
            dataset_split) + " and " + str(num_threads) + " extra threads per split...",
                           "TfRecordHandler:createTfRecords")

    def __writeTfrecord(self, data, mode, dataset_name, dataset_split):
        """
        Writes the tfrecord file for the given data.
        :param data: (Array of Dictionaries) The data to write in the tfrecord file
                       e.g.:[{"img": [1,2,3,4], "label": 1}, {"img": [1,2,3,4], "label": 1}].
        :param mode: (String) The mode of the saved record.
        :param dataset_name: (String) The name of the dataset.
        :param dataset_split: (Array) The splits of the dataset in format
                              [trainNum, evalNum, testNum, "prepreprocessing"].
        """
        with TfRecordExporter(self.__tfrecord_dir, dataset_name, dataset_split, mode, len(data)) as exporter:
            exporter.writeData(data)

    def __writeTfrecordBatchwise(self, dataset_provider, mode, dataset_name, dataset_split):
        """
        Writes the tfrecord file for the given data batchwise.
        :param dataset_provider: (DatasetProvider) The provider of the batches of the dataset.
        :param mode: (String) The mode of the saved record.
        :param dataset_name: (String) The name of the dataset.
        :param dataset_split: (Array) The splits of the dataset in format [trainNum, evalNum, testNum,
                                "prepreprocessing"].
        """
        num_read_in_batches = dataset_provider.getNumReadInBatches(mode)

        with TfRecordExporter(self.__tfrecord_dir, dataset_name, dataset_split, mode,
                              dataset_provider.getSetSize(mode)) as exporter:
            for i in range(0, dataset_provider.getNumReadInBatches(mode)):
                batch = dataset_provider.getNextReadInBatchInNumpy(mode)
                exporter.writeData(batch, False)

                self.__logger.info("Created tfrecord for batch " + str(i) + "/" + str(num_read_in_batches) + ": " +
                                   str(round(i / num_read_in_batches * 100, 2)) + "%.",
                                   "TfRecordHandler:createTfRecords")

    def __calculateThreadsForEachSplit(self, splits_to_write):
        """
        Calculates the threads per split an distributes the threads in two pairs,
        because one thread is like no thread ;).
        :param splits_to_write: (Integer) The number of splits to write.
        :returns threads_for_splits: (Array) The threads for each split.
        """
        threads_for_splits = np.full((splits_to_write), 0)
        not_used_threads_to_distribute = self.__num_threads - splits_to_write

        distribute_index = 0
        while not_used_threads_to_distribute >= 2:
            threads_for_splits[distribute_index] += 2
            distribute_index += 1
            not_used_threads_to_distribute -= 2
            if distribute_index >= (splits_to_write - 1):
                distribute_index = 0

        return threads_for_splits

