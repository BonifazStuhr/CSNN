import os
import glob
import fnmatch
import cv2

import numpy as np
from PIL import Image

from utils.functions import downloadDataset
from Input_Component.ADataProvider import ADataProvider

HOMEPAGE = "https://tiny-imagenet.herokuapp.com/"
DATA_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"

class TinyImageNetProvider(ADataProvider):
    """
    The TinyImageNetProvider reads the Tiny ImageNet and provides the dataset in various forms.
    The TinyImageNetProvider is not responsible for augmenting the dataset!
    :Attributes:
        __classIdToLabel:      (Dictionary) Contains the mapping from class id to label.
    """

    def __init__(self, train_size=100000, eval_size=0, test_size=10000, prepreporcessing_type=""'None'"",
                 data_shape=[3,64,64]):
        """
        Constructor, initialize member variables.
        :param train_size: (Integer) The size of the training set. 100000 by default.
        :param eval_size: (Integer) The size of the eval set. 0 by default.
        :param test_size: (Integer) The size of the test set. 10000 by default.
        :param prepreprocessing_type: (String) The type of the prepreprocessing before getting the dataset or writing
                                      the tfrecord. "'None'" by default.
        :param data_shape: (Array) The shape of a data_entry for the tfrecord. [3,64,64] by default.
        """
        super().__init__(dataset_path=os.path.join('data', 'TinyImageNet'),
                         dataset_name='TinyImageNet',
                         dataset_size=train_size+eval_size+test_size,
                         train_size=train_size,
                         eval_size=eval_size,
                         test_size=test_size,
                         prepreporcessing_type=prepreporcessing_type,
                         dataset_processable_at_once=True,
                         num_classes=200,
                         read_in_size=110000,
                         read_in_shape=[64, 64, 3],
                         tfrecord_shapes={"data": data_shape, "label": [200]},
                         tfrecord_datatyps={"data": "uint8", "label": "uint8"})

        self.__classIdToLabel = {}

    def loadDataset(self):
        """
        Reads and returns the dataset.
        :return: x_train: (Array) The train data.
        :return: y_train: (Array) The train label.
        :return: x_eval: (Array) The eval data.
        :return: y_eval: (Array) The eval label.
        :return: x_test: (Array) The test data.
        :return: y_test: (Array) The test label.
        """
        # Download Tiny ImageNet dataset if it not exists.
        if not os.path.exists(super()._getDatasetPath()):
            downloadDataset(DATA_URL, super()._getDatasetPath())

        # Reading both sources.
        image_train, labels_train = self.__readTinyImageNetToNumpy("train")
        images_test, labels_test = self.__readTinyImageNetToNumpy("test")

        return (image_train, labels_train), (None, None), (images_test, labels_test)

    def __readTinyImageNetToNumpy(self, dataset_part):
        """
        Reads, converts and returns the dataset_part of Tiny ImageNet in numpy format.
        :param dataset_part: (String) The string describing the dataset part.
        """
        images = []
        labels = []

        # To give each label an id
        label_id = 0

        # Read in labels
        if dataset_part is "train":
            image_files = []
            for root, dirnames, filenames in os.walk(super()._getDatasetPath() +'/tiny-imagenet-200/train'):
                for filename in fnmatch.filter(filenames, '*.JPEG'):
                    image_files.append(os.path.join(root, filename))
                    x = filename.split("_")[0]
                    if x not in self.__classIdToLabel.keys():
                        self.__classIdToLabel[x] = label_id
                        label_id += 1

        else:
            filenameToClassId = {}
            image_files = glob.glob(super()._getDatasetPath() + "/tiny-imagenet-200/val/images/*.JPEG")
            f = open(super()._getDatasetPath() +"/tiny-imagenet-200/val/val_annotations.txt", "r")
            for line in f.readlines():
                x = line.split("\t")
                filenameToClassId[x[0]] = x[1]


        # Read in images
        def imgToNp(image):
            (im_width, im_height) = image.size
            if image.mode is "L":
                image = np.array(image.getdata()).reshape((im_height, im_width, 1)).astype(np.uint8)
                return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

        for image_file_name in image_files:
            image = Image.open(image_file_name)
            image_np = imgToNp(image)
            images.append(image_np)

            if dataset_part is "train":
                labels.append(self.__classIdToLabel[image_file_name.split("/")[-1].split("_")[0]])
            else:
                labels.append(self.__classIdToLabel[filenameToClassId[image_file_name.split("/")[-1]]])

        return images, labels

    def getNumReadInBatches(self, mode):
        """
        Interface Method: Returns the number of batches for a given mode.
        :param mode: (String) The mode of the saved record.
        :returns num_batches: (Integer) The number of batches for the given mode.
        """
        # No need to implement this function for Tiny ImageNet, because Tiny ImageNet can be read at once.
        raise NotImplementedError('Not implemented for TinyImageNetProvider')

    def getNextReadInBatchInNumpy(self, mode):
        """
        Interface Method: Returns the next batch for the given mode.
        :param mode : (string) The mode of the saved record.
        :returns batch : (array of Dictionaries) The next batch of the dataset for the given mode in the form
        e.g.: {"data":(read_in_batch_size, 3, 64, 64), "label":(read_in_batch_size, ) or if onehot
        (read_in_batch_size, 200).
        """
        # No need to implement this function for Tiny ImageNet, because Tiny ImageNet can be read at once.
        raise NotImplementedError('Not implemented for TinyImageNetProvider')


