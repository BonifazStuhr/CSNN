import importlib
import os
import cv2
import tensorflow as tf
import numpy as np

from keras.preprocessing.image import ImageDataGenerator

from Input_Component.DataProviders.EncodingProvider import EncodingProvider
from Preprocessing_Component.Preprocessing.ZeroMeanUnitVarianceNormalizer import ZeroMeanUnitVarianceNormalizer
from Preprocessing_Component.Preprocessing.MaxDivNormalizer import MaxDivNormalizer
from Experiment_Component.Trainer.SingleGpuCsnnTrainer import SingleGpuCsnnTrainer
from Experiment_Component.Trainer.SingleGpuBackprobTrainer import SingleGpuBackprobTrainer
from Experiment_Component.Trainer.MultiGpuCsnnTrainer import MultiGpuCsnnTrainer
from Experiment_Component.Trainer.MultiGpuBackprobTrainer import MultiGpuBackprobTrainer
from Experiment_Component.Models.Csnn import Csnn
from Experiment_Component.Models.CsnnCnnHybrid import CsnnCnnHybrid
from Experiment_Component.ModelSuits.GeneratorModelSuits.CsnnGeneratorModelSuit import CsnnGeneratorModelSuit
from Experiment_Component.ModelSuits.TfRecordModelSuit.StandardTfRecordModelSuit import StandardTfRecordModelSuit
from Experiment_Component.ModelSuits.TfRecordModelSuit.FsModelSuit import FsModelSuit
from Experiment_Component.ModelSuits.ReconstructEncodingModelSuit import ReconstructEncodingModelSuit
from Experiment_Component.ModelSuits.GeneratorModelSuits.HybridGeneratorModelSuit import HybridGeneratorModelSuit
from Experiment_Component.Trainer.MultiGpuHybridTrainer import MultiGpuHybridTrainer

from Output_Component.TfRecord.TfRecordExporter import TfRecordExporter


def testSavedEncoding(encoding_name, encoding_shape, dataset_encoding, dataset_labels, label_shape,
                      encoding_config, test_batch_size, logger):
    """
    Tests the saved encoding with the given name with the given encoding and with the given dataset_labels.
    and saves the encoding to the experimentResults directory with the given model_name.
    :param encoding_name: (String) The name of the encoding to test.
    :param encoding_shape: (Array) The shape of the encoding to test.
    :param dataset_encoding: (Dictionary) The given encoding of the dataset
                             in form {"train:" ... , "eval":... "test": ...}
    :param dataset_labels: (Dictionary) The labels of the given encoding
                            in form {"train:" ... , "eval":... "test": ...}
    :param label_shape: (Array) The shape of the labels.
    :param encoding_config: (Dictionary) The configuration of the encoding to test.
    :param test_batch_size: (Integer) The batch size with with to compare the encodings.
    :param logger: (String) The logger to log the test.
    """

    logger.debug("Starting to test the saved encoding ...", "CsnnExperiment:testSavedEncoding")

    encoding_provider = EncodingProvider(encoding_name,
                                         encoding_shape,
                                         label_shape,
                                         encoding_config["splitOfDataset"][0],
                                         encoding_config["splitOfDataset"][1],
                                         encoding_config["splitOfDataset"][2],
                                         prepreporcessing_type=encoding_config["splitOfDataset"][3])
    # Start and init a Session
    sess = tf.Session()

    conf = encoding_config["tfRecordInputPipelineConfig"]["val"]
    conf["shuffleMB"] = 0

    # Build the input pipeline via the dataset provider.
    iterator, file_placeholder = \
        encoding_provider.getTfRecordInputPipelineIteratorsAndPlaceholdersFor("val", test_batch_size,
                                                                              conf)

    # Get the names of the input files to feed the pipeline later with train/eval/test files.
    dataset_files = encoding_provider.getTfRecordFileNames()

    for mode in ["train", "eval", "test"]:
        sess.run(iterator.initializer, feed_dict={file_placeholder: dataset_files[mode]})
        data_op, label_op = iterator.get_next()
        data, labels = sess.run((data_op, label_op))

        step = 0
        try:
            while True:
                batch_data, batch_labels = sess.run((data_op, label_op))
                labels = np.concatenate((labels, batch_labels))
                data = np.concatenate((data, batch_data))
                step += 1

        except tf.errors.OutOfRangeError:
            assert data.shape == dataset_encoding[mode].shape
            assert labels.shape == dataset_labels[mode].shape
            num_wrong_labels_direct_compare = 0
            num_wrong_labels = 0
            num_wrong_datasamples = 0
            num_wrong_labels_all_close = 0
            num_wrong_datasamples_all_close = 0
            for i in range(data.shape[0]):
                if np.argmax(labels[i]) != np.argmax(dataset_labels[mode][i]):
                    num_wrong_labels_direct_compare += 1
                if not np.allclose(labels[i], dataset_labels[mode][i], rtol=0, atol=0):
                    num_wrong_labels_all_close += 1
                if not np.allclose(data[i], dataset_encoding[mode][i], rtol=0, atol=0):
                    num_wrong_datasamples_all_close += 1
                if not np.array_equal(labels[i], dataset_labels[mode][i]):
                    num_wrong_labels += 1
                if not np.array_equal(data[i], dataset_encoding[mode][i]):
                    num_wrong_datasamples += 1
                # import cv2
                # cv2.imshow("in", np.reshape(data[i], [8, 8]))
                # cv2.imshow("out", np.reshape(dataset_encoding[mode][i], [8, 8]))
                # cv2.waitKey()
                # print(data[i])
                # print(dataset_encoding[mode][i])
                # print(data[i].shape)
                # print(dataset_encoding[mode][i].shape)

            logger.debug("num_wrong_labels_direct_compare: " + str(num_wrong_labels_direct_compare),
                         ":testSavedEncoding")
            logger.debug("num_wrong_labels_all_close: " + str(num_wrong_labels_all_close), ":testSavedEncoding")
            logger.debug("num_wrong_datasamples_all_close: " + str(num_wrong_datasamples_all_close),
                                ":testSavedEncoding")
            logger.debug("num_wrong_labels: " + str(num_wrong_labels), ":testSavedEncoding")
            logger.debug("num_wrong_datasamples: " + str(num_wrong_datasamples), ":testSavedEncoding")

        logger.debug("Finished to test the saved encoding ...", ":testSavedEncoding")
    sess.close()

def shuffelDatasetForXFold(data, xfold_seed, dataset_split):
    """
    Shuffels the given dataset with the given xfold_seed seed and returns the dataset with in form of the given split.
    :param data: (Dictionary) The dataset in the form {"x_train": ..., "y_train": ..., "x_eval":..., ...  }
    :param xfold_seed: (Integer) The seed for the shuffle operation.
    :param dataset_split: (array) The array containing the split numbers in the order:
                           [train_num, eval_num, test_num, "prepreprocessing_type"].
    :return: dataset: (Dictionary) The shuffled dataset in the given split.
    """
    train_size = dataset_split[0]
    eval_size = dataset_split[1]
    test_size = dataset_split[2]

    if (eval_size is 0) and (test_size is 0):
        input = data["x_train"]
        labels = data["y_train"]
    elif eval_size is 0:
        input = np.concatenate((data["x_train"], data["x_test"]), axis=0)
        labels = np.concatenate((data["y_train"], data["y_test"]), axis=0)
    else:
        input = np.concatenate((data["x_train"], data["x_eval"], data["x_test"]), axis=0)
        labels = np.concatenate((data["y_train"], data["y_eval"], data["y_test"]), axis=0)

    p = np.random.RandomState(seed=xfold_seed).permutation(len(labels))
    input = input[p]
    labels = labels[p]

    dataset = {
        "x_train": input[:train_size],
        "y_train": labels[:train_size],

        "x_eval": input[train_size:train_size + eval_size],
        "y_eval": labels[train_size:train_size + eval_size],

        "x_test": input[-test_size:],
        "y_test": labels[-test_size:],
    }
    return dataset

def getDatasetProvider(dataset_config):
    """
    Returns the DataProvider defined in the given config.
    :param dataset_config: (Dictionary) The config of the dataset for which to load the provider.
    :return: provider: (ADataProvider) The DataProvider defined in the given config.
    """
    dataset_name = dataset_config["nameOfDataset"]
    dataset_split = dataset_config["splitOfDataset"]

    provider_name = dataset_name + "Provider"

    # Dynamically import the experiment class by name.
    input_module = importlib.import_module("Input_Component.DataProviders." + provider_name)
    # Dynamically load the provider class by name.
    provider = getattr(input_module, provider_name)(train_size=dataset_split[0],
                                                    eval_size=dataset_split[1],
                                                    test_size=dataset_split[2])
    return provider

def prepareDataset(provider, dataset_config, xfold_seed=None, augment_data=False, normalize_data="zeroMeanUnitVariance",
                   rescale_input=None):
    """
    Prepares the given dataset for the experiment: If a xfold_seed is given the dataset will be shuffled.
    :param provider: (ADataProvider) The DataProvider to load the dataset.
    :param dataset_config: (Dictionary) The config of the dataset.
    :param xfold_seed: (Integer) The seed for the shuffle operation. None by default.
    :param augment_data: (Boolean) If True the data will be augmented as defined below. False by default.
    :param normalize_data: (Boolean) If True the data will be normalized. True by default.
    :param rescale_input: (Array) If set the input data will be rescaled. None by default.
    :return: dataset: (Dictionary) The shuffled dataset in the given split.
    :return: dataset_generator: (keras.preprocessing.image.ImageDataGenerator) The keras generator for the dataset.
    """
    dataset_split = dataset_config["splitOfDataset"]

    dataset = provider.getSplittedDatasetInNumpy()
    if xfold_seed:
        dataset = shuffelDatasetForXFold(dataset, xfold_seed, dataset_split)

    if normalize_data == "zeroMeanUnitVariance":
        dataset = ZeroMeanUnitVarianceNormalizer().preprocessingInNumpy(dataset)
    elif normalize_data == "maxDiv":
        dataset = MaxDivNormalizer(255.0).preprocessingInNumpy(dataset)

    if rescale_input:
        dataset = rescaleImages(dataset, rescale_input)

    if augment_data:
        dataset_generator = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
        )
    else:
        dataset_generator = ImageDataGenerator()

    dataset_generator.fit(dataset["x_train"])

    return dataset, dataset_generator

def rescaleImages(dataset, rescale_input):
    """
    Rescales the input of the given dataset to the given sizes.
    :param dataset: (Dictionary) The dataset to rescale.
    :param rescale_input: (Array) The new input sizes
    :return: dataset: (Dictionary) The rescaled dataset in the given split.
    """
    for mode in ["train", "eval", "test"]:
        resized_inputs = []
        inputs = dataset["x_"+mode]
        if (inputs.shape[1] == rescale_input[0]) and (inputs.shape[2] == rescale_input[1]):
            continue
        for input in inputs:
            resized_inputs.append(cv2.resize(input, dsize=(rescale_input[0], rescale_input[1]),
                                             interpolation=cv2.INTER_NEAREST))
        dataset["x_" + mode] = np.array(resized_inputs)
    return dataset

def trainAndValCsnn(csnn_config, dataset_generator, dataset, dataset_config, num_gpus, model_dir):
    """
    Trains and additionally evaluates the CSNN model defined in the csnn_config with the given dataset on num_gpus gpus
    and saves the model to the path model_dir.
    :param csnn_config: (Dictionary) The configuration of the CSNN containing all hyperparameters.
    :param dataset: (Dictionary) The dataset to train and val on.
    :param dataset_generator: (keras.preprocessing.image.ImageDataGenerator) The keras generator to train and val with.
    :param dataset_config: (Dictionary) The config of the dataset to train and val on.
    :param num_gpus: (Integer) The number of gpus to train and val with.
    :param model_dir: (String) The path to the directory in which the model is saved.
    """
    batch_size = csnn_config["batchSizes"][dataset_config["nameOfDataset"]]
    # Define the csnn trainer
    if (num_gpus <= 1) or (batch_size < num_gpus):
        csnn_trainer = SingleGpuCsnnTrainer()
    else:
        for l in range(0, len(csnn_config["layers"])):
            csnn_config["layers"][l]["verbose"] = 0
        csnn_trainer = MultiGpuCsnnTrainer(num_gpus)

    model_suit = CsnnGeneratorModelSuit(Csnn(csnn_config), dataset_generator, dataset, batch_size, csnn_trainer,
                                        num_gpus=num_gpus,
                                        y_shape=[None, dataset_config["numClasses"]],
                                        x_shape=[None, dataset_config["dataShape"][0], dataset_config["dataShape"][1],
                                                 dataset_config["dataShape"][2]],
                                        model_dir=model_dir)

    model_suit.doTraining(csnn_config["trainingSteps"], csnn_config["evalInterval"])

    if csnn_config["doVal"]:
        model_suit.doValidation("eval")

    model_suit.closeSession()

def prepareEncoding(csnn_config, dataset_generator, dataset, dataset_config, model_name, num_gpus, model_dir,
                    zero_mean_unit_variance=False, return_with_encoding=None):
    """
    Creates and prepares the encoding of the given dataset with the given CSNN model on num_gpus gpus
    and saves the encoding to the experimentResults directory with the given model_name.
    :param csnn_config: (Dictionary) The configuration of the CSNN containing all hyperparameters.
    :param dataset_generator: (keras.preprocessing.image.ImageDataGenerator) The keras generator to encode with.
    :param dataset: (Dictionary) The dataset to encode.
    :param dataset_config: (Dictionary) The config of the dataset to encode.
    :param model_name: (String) The name of the model to encode with.
    :param num_gpus: (Integer) The number of gpus to encode with.
    :param model_dir: (String) The path to the directory of the model (weights).
    :param zero_mean_unit_variance: (Boolean) If true the encoding will be saved with zero mean and unit variance.
                                     False by default.
    :param return_with_encoding: (Dictionary) If set the encoding will be returned with given the dataset.
                                 None by default.
    :return: provider: (EncodingProvider) The DataProvider for the encoding created trough the CSNN.
    :return: encoding_out: (Dictionary) Optional. The encoding. E.g. {"x_train": ...}.
    """
    batch_size = csnn_config["batchSizeForEncoding"][dataset_config["nameOfDataset"]]

    # Load the CSNN model .
    if (num_gpus <= 1) or (batch_size < num_gpus):
        csnn_trainer = SingleGpuCsnnTrainer()
    else:
        csnn_trainer = MultiGpuCsnnTrainer(num_gpus)

    for l in range(0, len(csnn_config["layers"])):
        csnn_config["layers"][l]["verbose"] = 0

    model_suit = CsnnGeneratorModelSuit(Csnn(csnn_config), dataset_generator, dataset, batch_size, csnn_trainer,
                                        num_gpus=num_gpus,
                                        y_shape=[None, dataset_config["numClasses"]],
                                        x_shape=[None, dataset_config["dataShape"][0], dataset_config["dataShape"][1],
                                                 dataset_config["dataShape"][2]],
                                        model_dir=model_dir)

    # Create the encoded dataset.
    train_encoding_mean = None
    train_encoding_variance = None

    encoding_out = {"x_train": [], "y_train": [], "i_train": [], "x_eval": [], "y_eval": [], "i_eval": [],
                    "x_test": [], "y_test": [], "i_test": []}

    # For each subset:
    for mode in ["train", "eval", "test"]:

        # create the encoding: If the encoding size equals the subdataset encoding size, encode the entire subdataset,
        # else generate the encoding for the given size with the generator.
        encoding, labels, input = model_suit.createEncoding(mode,
                                                 csnn_config["encodingSizes"][dataset_config["nameOfDataset"]][mode],
                                                            return_with_encoding=return_with_encoding)
        # calculate the variance in the encoding
        if zero_mean_unit_variance:
            if mode is "train":
                train_encoding_mean = np.mean(encoding)
                train_encoding_variance = np.std(encoding)
            encoding = (encoding - train_encoding_mean) / (train_encoding_variance + 1e-7)

        # get the shapes of label and encodings
        encoding_shape = np.array(encoding[0][0][0]).shape
        shape = str(encoding_shape[0]) + str(encoding_shape[1]) + str(encoding_shape[2])
        label_shape = np.array(labels[0][0][0]).shape

        # name the encoding and create the directory
        encoding_name = "Csnn" + dataset_config["nameOfDataset"] + shape + "Encoding" + str(model_name)
        splitOfDataset = dataset_config["splitOfDataset"]
        splitOfDataset[3] = splitOfDataset[3].replace("'", "")
        if not os.path.exists("data/" + encoding_name):
            os.makedirs("data/" + encoding_name)

        # and save the encoding as TfRecord.
        with TfRecordExporter("data", encoding_name, dataset_config["splitOfDataset"], mode) as exporter:
            # Therefor save each batch
            for batch_num in range(0, len(encoding)):
                # from each gpu
                for gpu in range(0, len(encoding[batch_num])):
                    label_batch = labels[batch_num][gpu].astype(np.uint8)
                    # to the file.
                    exporter.writeData({"data": encoding[batch_num][gpu], "label": label_batch}, shuffle=False)
                    if return_with_encoding:
                        for s in range(0, len(encoding[batch_num][gpu])):
                            encoding_out["x_" + mode].append(encoding[batch_num][gpu][s])
                            encoding_out["y_" + mode].append(label_batch[s])
                            encoding_out["i_" + mode].append(input[batch_num][gpu][s])
        if return_with_encoding:
            encoding_out["x_" + mode] = np.array(encoding_out["x_" + mode])
            encoding_out["y_" + mode] = np.array(encoding_out["y_" + mode])
            encoding_out["i_" + mode] = np.array(encoding_out["i_" + mode]+abs(np.min(encoding_out["i_" + mode])))
            encoding_out["i_" + mode] = np.divide(encoding_out["i_" + mode], np.max(encoding_out["i_" + mode]))

    model_suit.closeSession()

    #encoding_name = "Csnn" + dataset_config["nameOfDataset"] + "44768" + "Encoding" + str(csnn_config["modelName"])
    #encoding_shape = [4, 4, 768]
    #label_shape = [10]

    encoding_provider = EncodingProvider(encoding_name,
                                         encoding_shape,
                                         label_shape,
                                         dataset_config["splitOfDataset"][0],
                                         dataset_config["splitOfDataset"][1],
                                         dataset_config["splitOfDataset"][2],
                                         prepreporcessing_type="'"+dataset_config["splitOfDataset"][3]+"'")
                                         #prepreporcessing_type = dataset_config["splitOfDataset"][3])
    if return_with_encoding:
        return encoding_provider, encoding_out

    return encoding_provider


def trainAndValClassifier(model, model_config, encoding_provider, encoding_config, num_gpus, model_dir):
    """
    Trains and additionally evaluates and tests the classifier defined in the model_config with the given encoding on
    num_gpus gpus and saves the model to the path model_dir.
    :param model: ("Model") The classifier to train, eval and test.
    :param model_config: (Dictionary) The configuration of the classifier containing all hyperparameters.
    :param encoding_provider: (EncodingProvider) The DataProvider for the encoding.
    :param encoding_config: (Dictionary) The config of the encoding to train, eval and test on.
    :param num_gpus: (Integer) The number of gpus to train, eval and test with.
    :param model_dir: (String) The path to the directory in which the classifier is saved.
    """
    # Train the classifier.
    if num_gpus <= 1:
        mlp_trainer = SingleGpuBackprobTrainer()
    else:
        mlp_trainer = MultiGpuBackprobTrainer(num_gpus)

    model_suit = StandardTfRecordModelSuit(model, encoding_provider,
                                   encoding_config["tfRecordInputPipelineConfig"], model_config["batchSize"],
                                   mlp_trainer, model_dir=model_dir)
    model_suit.doTraining(model_config["trainingSteps"], model_config["evalInterval"], only_save_best_checkpoints=True)
    model_suit.closeSession()

    # Load the best checkpoint
    model_suit = StandardTfRecordModelSuit(model, encoding_provider,
                                   encoding_config["tfRecordInputPipelineConfig"], model_config["batchSize"],
                                   mlp_trainer, model_dir=model_dir)

    # Evaluate and test the best checkpoint.
    model_suit.doDatesetValidation()
    model_suit.closeSession()

def trainAndValFewShotClassifier(model, model_config, encoding_provider, encoding_config, num_gpus, model_dir):
    """
    Trains and additionally evaluates and tests the few-shot classifier defined in the model_config with the given
    encoding on num_gpus gpus and saves the model to the path model_dir.
    :param model: ("Model") The few-shot classifier to train, eval and test.
    :param model_config: (Dictionary) The configuration of the few-shot classifier containing all hyperparameters.
    :param encoding_provider: (EncodingProvider) The DataProvider for the encoding.
    :param encoding_config: (Dictionary) The config of the encoding to train, eval and test on.
    :param num_gpus: (Integer) The number of gpus to eval and test with.
    :param model_dir: (String) The path to the directory in which the few-shot classifier is saved.
    """
    for shot_size in range(model_config["minShot"], model_config["maxShot"]+1,  model_config["shotIncrementation"]):
        mlp_trainer = SingleGpuBackprobTrainer()

        model_suit = FsModelSuit(model, encoding_provider, encoding_config["tfRecordInputPipelineConfig"],
                                 model_config["batchSize"], mlp_trainer, model_dir=model_dir + str(shot_size) + "Shot")
        model_suit.doTraining(shot_size, shot_training_steps=model_config["shotTrainingSteps"],
                              eval_interval=model_config["evalInterval"])
        model_suit.closeSession()

        # Load the best checkpoint
        model_suit = FsModelSuit(model, encoding_provider, encoding_config["tfRecordInputPipelineConfig"],
                                 model_config["batchSize"], mlp_trainer, model_dir=model_dir + str(shot_size) + "Shot")

        # Evaluate and test the best checkpoint.
        model_suit.doDatesetValidation()
        model_suit.closeSession()


def trainAndValReconstructionModel(model, model_config, num_gpus, model_dir, dataset, dataset_config):
    """
    Trains and additionally evaluates and tests the reconstruction model defined in the model_config with the given
    encoding on num_gpus gpus and saves the model to the path model_dir.
    :param model: ("Model") The reconstruction model  to train, eval and test.
    :param model_config: (Dictionary) The configuration of the reconstruction model containing all hyperparameters.
    :param num_gpus: (Integer) The number of gpus to train, eval and test with.
    :param model_dir: (String) The path to the directory in which the reconstruction model is saved.
    :param: dataset: (Dictionary) The shuffled dataset in the given split.
    :param: dataset_config: (Dictionary) The config of the dataset to encode.
    """
    num_gpus = 1
    rec_trainer = SingleGpuBackprobTrainer()

    model_suit = ReconstructEncodingModelSuit(model, dataset, model_config["batchSize"], rec_trainer,
                                              num_gpus=num_gpus,
                                              x_shape=[None, dataset["x_train"].shape[1],
                                                       dataset["x_train"].shape[2],
                                                       dataset["x_train"].shape[3]],
                                              y_shape=[None, dataset_config["dataShape"][0],
                                                       dataset_config["dataShape"][1],
                                                       dataset_config["dataShape"][2]], model_dir=model_dir)

    model_suit.doTraining(model_config["trainingSteps"], model_config["evalInterval"], only_save_best_checkpoints=True)
    model_suit.closeSession()

    # Load the best checkpoint
    model_suit = ReconstructEncodingModelSuit(model, dataset, model_config["batchSize"], rec_trainer, num_gpus=num_gpus,
                                              x_shape=[None, dataset["x_train"].shape[1],
                                                       dataset["x_train"].shape[2],
                                                       dataset["x_train"].shape[3]],
                                              y_shape=[None, dataset_config["dataShape"][0],
                                                       dataset_config["dataShape"][1],
                                                       dataset_config["dataShape"][2]], model_dir=model_dir)

    # Evaluate and test the best checkpoint.
    model_suit.doDatesetValidation()
    model_suit.saveReconstructions("test")
    model_suit.closeSession()

def trainAndValHybrid(csnn_config, dataset_generator, dataset, dataset_config, num_gpus, model_dir):
    """
    Trains and additionally evaluates the CSNN model defined in the csnn_config with the given dataset on num_gpus gpus
    and saves the model to the path model_dir.
    :param csnn_config: (Dictionary) The configuration of the CSNN containing all hyperparameters.
    :param dataset: (Dictionary) The dataset to train and val on.
    :param dataset_generator: (keras.preprocessing.image.ImageDataGenerator) The keras generator to train and val with.
    :param dataset_config: (Dictionary) The config of the dataset to train and val on.
    :param num_gpus: (Integer) The number of gpus to train and val with.
    :param model_dir: (String) The path to the directory in which the model is saved.
    """
    batch_size = csnn_config["batchSizes"][dataset_config["nameOfDataset"]]
    # Define the csnn trainer
    if (num_gpus <= 1) or (batch_size < num_gpus):
        print("HYBRID TRAINING IS NOT IMPLEMENTED FOR ONE GPU")
        return
    else:
        for l in range(0, len(csnn_config["layers"])):
            csnn_config["layers"][l]["verbose"] = 0
        csnn_trainer = MultiGpuHybridTrainer(num_gpus)

    csnn_config["numGpus"] = num_gpus
    csnn_config["numClasses"] = dataset_config["numClasses"]
    model_suit = HybridGeneratorModelSuit(CsnnCnnHybrid(csnn_config), dataset_generator, dataset, batch_size, csnn_trainer,
                                        num_gpus=num_gpus,
                                        y_shape=[None, dataset_config["numClasses"]],
                                        x_shape=[None, dataset_config["dataShape"][0], dataset_config["dataShape"][1],
                                                 dataset_config["dataShape"][2]],
                                        model_dir=model_dir)

    model_suit.doTraining(csnn_config["trainingSteps"], csnn_config["evalInterval"])

    if csnn_config["doVal"]:
        model_suit.doValidation("eval")

    model_suit.closeSession()