import tensorflow as tf
import numpy as np

class TfRecordProvider:
    """
    The TfRecordProvider provides the optimized input pipeline for the tfrecords.

    :Attributes:
        __tf_record_datatypes:         (Dictionary) The datatyps of a entry in the tfrecord
                                        e.g. {"data":"float32", "label":"uint8"}.
        __tf_datatyps_after_parse:     (Dictionary) The datatyps of a entry after it is parsed
                                        e.g. {"data":"tf.float32", "label":"uint8"}.
        __input_pipeline_config:       (Dictionary) The configuration of the input pipeline from the experiment
                                        config.
        __tfrecord_file:               (String) The path to the tfrecord file.
        __tfrecord_dataset:            (tf.data.TFRecordDataset) The tf class to read in and convert the tfrecord
                                        into a TFRecordDataset.
        __tfrecord_shape_lens:         (Dictionary) The shapes of a entry in the tfrecord file
                                        e.g. {"data":[1,28,28], "label":[10]}.
        __tfrecord_shapes:             (Dictionary) The shapes of a entry in the tfrecord file
                                        e.g. {"data":[1,28,28], "label":[10]}.
    """
    def __init__(self):
        """
        Constructor, initialize member variables.
        """
        # Set Datatype.
        self.__tf_record_datatypes = None
        self.__tf_datatyps_after_parse = None

        # Config
        self.__input_pipeline_config = None

        # Tfrecord file path
        self.__tfrecord_file = None
        self.__tfrecord_dataset = None

        # Tfrecord shape lens
        self.__tfrecord_shape_lens = None

        # Tfrecord shapes
        self.__tfrecord_shapes = None

    def getInputPipelineFrom(self, input_pipeline_config, batch_size, tfrecord_shapes, tfrecord_datatyps,\
                            tf_datatyps_after_parse={"data": tf.float32, "label": tf.uint8}, repeat=True):
        """
        To get the input pipeline for the given tfrecod file with the given config.
        :param input_pipeline_config: (Dictionary) The configuration of the input pipline from the experiment config.
        :param batch_size: (Integer) The batch size for the returned data of the pipeline.
        :param tfrecord_shapes: (Dictionary) The shapes of a entry in the tfrecord file
                                e.g. {"data":[1,28,28], "label":[10]}.
        :param tfrecord_datatyps: (Dictionary) The datatyps of a entry in the tfrecord
                                  e.g. {"data":"float32", "label":"uint8"}.
        :param tf_datatyps_after_parse: (Dictionary) The datatyps of a entry after it is parsed
                                        e.g. {"data":"tf.float32", "label":"uint8"}.
        :param repeat: (Boolean) If true, the dataset will be repeated if iterator iterated over whole dataset
                                (for training).
        :return: tfrecord_dataset: (Array) The Datataset with the entry_parts in order e.g. [data, label].
        :return: tfrecord_files_placeholder: (tf.placeholder) The placeholder for the file path input to the pipeline.
        """
        with tf.name_scope('dataset'), tf.device('/cpu:0'):

            # Prepare needed params for a controlled dataset read in.
            self.__tfrecord_shapes = tfrecord_shapes
            self.__tf_datatyps_after_parse = tf_datatyps_after_parse

            self.__tfrecord_shape_lens = {}
            self.__tf_record_datatypes = {}
            bytes_per_item = 0
            for entry_part_name, shape, datatype in zip(tfrecord_shapes.keys(), tfrecord_shapes.values(),
                                                        tfrecord_datatyps.values()):
                self.__tfrecord_shape_lens[entry_part_name] = len(shape)
                self.__tf_record_datatypes[entry_part_name] = getattr(tf, datatype)

                # Compute needed bytes per item for optimisation
                bytes_per_item += np.prod(shape) * np.dtype(datatype).itemsize

            # Set config
            self.__input_pipeline_config = input_pipeline_config

            # Set tfrecord file path
            self.__tfrecord_files_placeholder = tf.placeholder(tf.string, shape=[None])
            self.__tfrecord_dataset = tf.data.TFRecordDataset(self.__tfrecord_files_placeholder, compression_type='',
                                buffer_size=self.__input_pipeline_config["bufferMB"] << 20) # Converting MB into Bytes

            # Furthermore, if your batch size is in the hundreds or thousands, your pipeline will likely additionally
            # benefit from parallelizing the batch creation. To this end, the tf.data API provides the tf.contrib.data.
            # map_and_batch transformation, which effectively "fuses" the map and batch transformations.
            # TODO: Apply mapAndBatch when needed.
            #if self.__input_pipeline_config["mapAndBatch"]:
             #   self.__tfrecord_dataset = self.__tfrecord_dataset.apply(tf.contrib.data.map_and_batch(
              #      map_func=self.__parseTfRecord, batch_size=batch_size,
               #     num_parallel_calls=self.__input_pipeline_config["numThreads"]))
            # Invoking the user-defined function passed into the map transformation has overhead related to scheduling
            # and executing the user-defined function. Normally, this overhead is small compared to the amount of
            # computation performed by the function. However, if map does little work, this overhead can dominate the
            # total cost. In such cases, we recommend vectorizing the user-defined function (that is, have it operate
            # over a batch of inputs at once) and apply the batch transformation before the map transformation.
            # TODO: Apply batch befor map when needed.
            self.__tfrecord_dataset = self.__tfrecord_dataset.map(self.__parseTfRecord,
                                                                  num_parallel_calls=self.__input_pipeline_config[
                                                                      "numThreads"])

            # Todo: if num workers is higher then 1
            # Creates a Dataset that includes only 1/num_shards of this dataset.
            # This dataset operator is very useful when running distributed training, as it allows each worker to read a
            # unique subset.
            # Important caveats:
            # Be sure to shard before you use any randomizing operator (such as shuffle).
            # Generally it is best if the shard operator is used early in the dataset pipeline.
            # For example, when reading from a set of TFRecord files, shard before converting the dataset to input
            # samples. This avoids reading every file on every worker. The following is an example of an efficient
            # sharding strategy within a complete pipeline:

            # self.__tfrecord_dataset = self.__tfrecord_dataset.shard(FLAGS.num_workers, FLAGS.worker_index)

            # The tf.data.Dataset.cache transformation can cache a dataset, either in memory or on local storage.
            # If the user-defined function passed into the map transformation is expensive, apply the cache
            # transformation after the map transformation as long as the resulting dataset can still fit into memory or
            # local storage. If the user-defined function increases the space required to store the dataset beyond the
            # cache capacity, consider pre-processing your data before your training job to reduce resource usage.
            if self.__input_pipeline_config["cache"]:
                self.__tfrecord_dataset = self.__tfrecord_dataset.cache()

            # If the repeat transformation is applied before the shuffle transformation, then the epoch boundaries are
            # blurred. That is, certain elements can be repeated before other elements appear even once.
            # On the other hand, if the shuffle transformation is applied before the repeat transformation, then
            # performance might slow down at the beginning of each epoch related to initialization of the internal state
            # of the shuffle transformation. In other words, the former (repeat before shuffle) provides better
            # performance, while the latter (shuffle before repeat) provides stronger ordering guarantees.
            #
            # When possible, we recommend using the fused tf.contrib.data.shuffle_and_repeat transformation,
            # which combines the best of both worlds (good performance and strong ordering guarantees).
            # Otherwise, we recommend shuffling before repeating.

            if self.__input_pipeline_config["shuffelAndRepeat"] and repeat:
                # Converting MB into Bytes and dividing  (//=floor division) the avaible bits by the
                # bytes_per_item, to get maximum possible items. Avoiding 0 values.
                self.__tfrecord_dataset = self.__tfrecord_dataset.apply(tf.contrib.data.shuffle_and_repeat(
                    ((self.__input_pipeline_config["shuffleMB"] << 20) - 1) // bytes_per_item + 1,seed=42))
            else:
                # The tf.data.Dataset.shuffle transformation randomizes the order of the dataset's examples.
                if self.__input_pipeline_config["shuffleMB"] > 0:
                    # Converting MB into Bytes and dividing  (//=floor division) the avaible bits by the
                    # bytes_per_item, to get maximum possible items. Avoiding 0 values.
                    self.__tfrecord_dataset = self.__tfrecord_dataset.shuffle(
                        ((self.__input_pipeline_config["shuffleMB"] << 20) - 1) // bytes_per_item + 1, seed=42)

                # The tf.data.Dataset.repeat transformation repeats the input data a finite (or infinite) number of
                # times; each repetition of the data is typically referred to as an epoch.
                if repeat:
                    self.__tfrecord_dataset = self.__tfrecord_dataset.repeat()

            # Use the prefetch transformation to overlap the work of a producer and consumer. In particular, we
            # recommend adding prefetch(n) (where n is the number of elements / batches consumed by a training step)
            # to the end of your input pipeline to overlap the transformations performed on the CPU with the training
            # done on the accelerator.
            if self.__input_pipeline_config["prefetchMB"] > 0:
                # Converting MB into Bytes and dividing  (//=floor division) the avaible bits by the
                # bytes_per_item, to get maximum possible items. Avoiding 0 values.
                self.__tfrecord_dataset = self.__tfrecord_dataset.prefetch(
                    ((self.__input_pipeline_config["prefetchMB"] << 20) - 1) // bytes_per_item + 1)

            # Batch the dataset.
            self.__tfrecord_dataset = self.__tfrecord_dataset.batch(batch_size=batch_size)

            return self.__tfrecord_dataset, self.__tfrecord_files_placeholder

    def __parseTfRecord(self, tfrecord):
        """
        Parses the given tfrecord and returns its data.
        :param tfrecord: (Tensor) The tfrecord to parse.
        :return: parsed_data: (Tensor) The parsed data in the tfrecord.
        """
        features = tf.parse_single_example(tfrecord, features={
            'data': tf.FixedLenFeature([], tf.string),
            'dataShape': tf.FixedLenFeature([self.__tfrecord_shape_lens["data"]], tf.int64),
            'label': tf.FixedLenFeature([], tf.string),
            'labelShape': tf.FixedLenFeature([self.__tfrecord_shape_lens["label"]], tf.int64),
        })
        data = tf.decode_raw(features['data'], self.__tf_record_datatypes["data"])
        data = tf.reshape(data, features['dataShape'])
        data.set_shape(self.__tfrecord_shapes["data"])
        data = tf.cast(data, self.__tf_datatyps_after_parse["data"])

        label = tf.decode_raw(features['label'], self.__tf_record_datatypes["label"])
        label = tf.reshape(label, features['labelShape'])
        label.set_shape(self.__tfrecord_shapes["label"])
        label = tf.cast(label, self.__tf_datatyps_after_parse["label"])

        return [data, label]

