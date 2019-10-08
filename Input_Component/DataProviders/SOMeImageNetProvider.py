import os
import tarfile
import cv2
import numpy as np
import xml.etree.ElementTree as ET

from Input_Component.ADataProvider import ADataProvider

class SOMeImageNetProvider(ADataProvider):
    """
    The SOMeImageNetProvider reads a subset of the ImageNet dataset and provides the dataset in various forms.
    The SOMeImageNetProvider is not responsible for augmenting the dataset!
    """

    def __init__(self, train_size=None, eval_size=None, test_size=None, prepreporcessing_type=""'None'"",
                 data_shape=[3,256,256]):
        """
        Constructor, initialize member variables.
        :param train_size: (Integer) The size of the training set. None by default.
        :param eval_size: (Integer) The size of the eval set. None by default.
        :param test_size: (Integer) The size of the test set. None by default.
        :param prepreprocessing_type: (String) The type of the prepreprocessing before getting the dataset or writing
                                       the tfrecord. "'None'" by default.
        :param data_shape: (Array) The shape of a data_entry for the tfrecord. [3 256,256] by default.
        """
        project_root_path = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
        self.__dataset_path = os.path.split(project_root_path)[0] + "/data/ImageNet/"

        self.__classes = self.__readSubsetClasses(self.__dataset_path)
        self.__num_classes = len(self.__classes)
        self.__id_to_name, self.__id_to_class_number = self.__readIdMapping(self.__dataset_path, self.__classes)

        self.__read_in_w = 256
        self.__read_in_h = 256

        #print(self.__id_to_name)
        #print(self.__num_classes)
        #print(self.__classes)

        #Read in file paths of subclasses
        self.__image_members, self.__label_members, self.__samples_per_label, _, _, _ = \
            self.__readSOMeImageNetMembers()


        super().__init__(dataset_path=self.__dataset_path,
                         dataset_name='SOMeImageNet',
                         dataset_size=train_size+eval_size+test_size,
                         train_size=train_size,
                         eval_size=eval_size,
                         test_size=test_size,
                         prepreporcessing_type=prepreporcessing_type,
                         dataset_processable_at_once=True,
                         num_classes=len(self.__classes),
                         read_in_size=train_size+eval_size+test_size,
                         read_in_shape=[self.__read_in_h, self.__read_in_w, 3],
                         tfrecord_shapes={"data":data_shape, "label":[len(self.__classes)]},
                         tfrecord_datatyps={"data":"uint8", "label":"uint8"})


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
        # Reading both sources.
        image_train, labels_train = self.__readSOMeImageNetToNumpy("train")
        image_eval, labels_eval = self.__readSOMeImageNetToNumpy("eval")
        #images_test, labels_test = self.__readSOMeImageNetToNumpy("test")

        return (image_train, labels_train), (None, None), (image_eval, labels_eval)

    def __readSOMeImageNetToNumpy(self, mode):
        """
        Reads, converts and returns the dataset_part of Tiny SOMeImageNetProvider in numpy format.
        :param mode: (String) The mode of the saved record.
        :return: images: (np.array) The images in (datasetpart_size, 3, 256, 256) shape.
        :return: labels: (np.array) The labels in (datasetpart_size,) shape.
        """
        tar_archive = tarfile.open(self.__dataset_path + "imagenet_object_localization.tar", "r")
        images = []
        labels = []
        for i in range(0, len(self.__label_members[mode])):
            img_tar_info = self.__image_members[mode][i]
            label = self.__label_members[mode][i]

            if label is "None":
                continue
            else:
                labels.append(self.__id_to_class_number[label])

            # read and convert image
            jpeg_file = tar_archive.extractfile(img_tar_info)
            content = jpeg_file.read()
            img = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
            img = cv2.resize(img, dsize=(self.__read_in_h, self.__read_in_w), interpolation=cv2.INTER_LINEAR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)

        images = np.reshape(images, [len(images), self.__read_in_h, self.__read_in_w, 3])
        labels = np.array(labels)

        return images, labels


    def __readSOMeImageNetMembers(self):
        """
        Reads the members of the subset from the compressed file and returns them.
        :return: image_members: ("Member") The image members in the zip file.
        :return: label_members: ("Member") The label members in the zip file.
        :return: samples_per_label: (Array) The numer of samples per label.
        :return: train_size: (Integer) The number of training samples.
        :return: eval_size: (Integer) The number of eval samples.
        :return: test_size: (Integer) The number of test samples.
        """
        # read archive
        tar_archive = tarfile.open(self.__dataset_path + "imagenet_object_localization.tar", "r")
        archive_members = tar_archive.getmembers()

        in_img_members = {"train": [], "eval": [], "test": []}
        in_label_members = {"train": [], "eval": [], "test": []}
        for member in archive_members:
            if "ILSVRC/Data/CLS-LOC/train/" in member.name and ".JPEG" in member.name:
                in_img_members["train"].append(member)
            elif "ILSVRC/Data/CLS-LOC/val/" in member.name and ".JPEG" in member.name:
                in_img_members["eval"].append(member)
            #elif "ILSVRC/Data/CLS-LOC/test/" in member.name and ".JPEG" in member.name:
            #    self.__in_img_members["test"].append(member)
            # elif "ILSVRC/Annotations/CLS-LOC/train/" in member.name and ".xml" in member.name:
            #    self.__label_members["train"].append(member)
            elif "ILSVRC/Annotations/CLS-LOC/val/" in member.name and ".xml" in member.name:
                in_label_members["eval"].append(member)
            # elif "ILSVRC/Annotations/CLS-LOC/test/" in member.name and ".xml" in member.name:
            #    self.__label_members["test"].append(member)

        in_img_members["eval"].sort(key=lambda x: x.name.split("_")[-1].split(".")[0])
        in_label_members["eval"].sort(key=lambda x: x.name.split("_")[-1].split(".")[0])
        p = np.random.RandomState(seed=42).permutation(len(in_img_members["train"]))
        in_img_members["train"] = np.array(in_img_members["train"])[p]
        p = np.random.RandomState(seed=42).permutation(len(in_img_members["eval"]))
        in_img_members["eval"] = np.array(in_img_members["eval"])[p]
        in_label_members["eval"] = np.array(in_label_members["eval"])[p]

        #p = np.random.RandomState(seed=42).permutation(len(self.__in_img_members["test"]))
        #self.__in_img_members["test"] = np.array(self.__in_img_members["test"])[p]

        label_members = {"train": [], "eval": [], "test": []}
        image_members = {"train": [], "eval": [], "test": []}
        samples_per_label = {label: 0 for label in self.__classes}
        for mode in ["train", "eval", "test"]:
            for i in range(len(in_img_members[mode])):
                img_tar_info = in_img_members[mode][i]

                label = None
                if mode is "eval":
                    # get files.
                    lable_tar_info = in_label_members[mode][i]
                    xml_file = tar_archive.extractfile(lable_tar_info)
                    tree = ET.parse(xml_file)
                    label = tree.find("./object/name").text
                elif mode is "train":
                    label = img_tar_info.name.split("/")[4]
                #elif mode is "test":
                #    label = "None"  # Dataset from Kaggle contains no test labels because of the competition.

                if label in self.__classes:
                    label_members[mode].append(label)
                    image_members[mode].append(in_img_members[mode][i])
                    samples_per_label[str(label)] += 1

        return image_members, label_members, samples_per_label, len(label_members["train"]), \
               len(label_members["eval"]), len(label_members["test"])


    def __readIdMapping(self, synset_mapping_path, classes):
        """
        Reads the mapping from the class id to the class name and class number from the file "LOC_synset_mapping.txt".
        :param synset_mapping_path: (String) The path to "LOC_synset_mapping.txt".
        :param classes: (Array) The class ids to read the mapping for.
        :return: id_to_name: (Dictionary) The id to class mapping. Key: id, Value: class.
        :return: id_to_class_number: (Dictionary) The id to class number mapping. Key: id, Value: class number.
        """
        synset_mapping = open(synset_mapping_path + "LOC_synset_mapping.txt", "r")
        content = synset_mapping.read()
        content_lines = content.splitlines()

        id_to_name = {}
        id_to_class_number = {}
        class_number = 0
        for line in content_lines:
            id_first_name_rest = line.split(" ", 1)
            if id_first_name_rest[0] in classes:
                if id_first_name_rest[0] not in id_to_name.keys():
                    id_to_name[id_first_name_rest[0]] = id_first_name_rest[1]
                    id_to_class_number[id_first_name_rest[0]] = class_number
                    class_number += 1

        return id_to_name, id_to_class_number

    def __readSubsetClasses(self, subset_file_path):
        """
        Reads the classes to read in from text file imagenet_subset_classes.txt. These defines the subset.
        :param subset_file_path: (String) The path to imagenet_subset_classes.txt.
        :return: imagenet_subset_classes: (Array) The class ids of the subset from the imagenet_subset_classes.txt file.
        """
        classes = open(subset_file_path + "imagenet_subset_classes.txt", "r")
        classes = classes.read()
        imagenet_subset_classes = classes.splitlines()
        return imagenet_subset_classes

    def getNumReadInBatches(self, mode):
        """
        Interface Method: Returns the number of batches for a given mode.
        :param mode : (String) The mode of the saved record.
        :return: num_batches : (Integer) The number of batches for the given mode.
        """
        # No need to implement this function for SOMeImageNet, because SOMeImageNet can be read at once.
        raise NotImplementedError('Not implemented for SOMeImageNetProvider')

    def getNextReadInBatchInNumpy(self, mode):
        """
        Interface Methode: Returns the next batch for the given mode.
        :param mode : (String) The mode of the saved record.
        :returns batch : (Array of Dictionaries) The next batch of the dataset for the given mode in the form
        e.g.: {"data":(read_in_batch_size, 3, 256, 256), "label":(read_in_batch_size, ) or if onehot
        (read_in_batch_size,num_classes).
        """
        # No need to implement this function for SOMeImageNet, because Tiny ImageNet can be read at once.
        raise NotImplementedError('Not implemented for SOMeImageNetProvider')
