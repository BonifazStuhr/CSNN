import cv2
import numpy as np

from LoggerNames import LoggerNames
from Logger_Component.SLoggerHandler import SLoggerHandler

class ImagePlusLabelVisualizer:
    """
    The ImagePlusLabelVisualizer tries to visualize the images and labels via cv2 and logs additional information,
    like the shape of the images and labels on screen.

    :Attributes:
        __logger:  (Logger) The logger for the visualizer.
    """

    def __init__(self):
        """
        Constructor, initialize member variables.
        """
        self.__logger = SLoggerHandler().getLogger(LoggerNames.LOGGER_C)

    def visualizeImagesAndLabelsWithBreak(self, images, labels, indexes=[0]):
        """
        Visualizes the images and labels defined in the indexes variable.
        Prints additional information on screen.
        Stops the program until some input is given.
        :param images: (np.array) The images to visualize.
        :param labels: (np.array) The labels to visualize.
        :param indexes: (Array) The indexes of the images and labels to visualize.
        """
        self.logImagesAndLabelsInfo(images, labels)

        for index in indexes:
            self.visualizeImageAndLabelWithBreak(images[index], labels[index])

    def visualizeImageAndLabelWithBreak(self, image, label):
        """
        Visualizes the image and label given.
        Prints additional information on screen.
        Stops the program until some input is given.
        :param images: (np.array) The image to visualize.
        :param labels: (np.array) The label to visualize.
        """
        self.logImagesAndLabelsInfo(image, label)
        self.__visualizeImageAndLabelWithBreak(image, label)

    def logImagesAndLabelsInfo(self, image, label):
        """
        Prints additional information of the image and label like the shape on screen.
        :param images: (np.array) The image to visualize.
        :param labels: (np.array) The label to visualize.
        """
        info_text = "\n*********ImagePlusLabelVisualizer*********\nImages shape: " + str(image.shape) + "\n" + \
                    "Labels shape: " + str(label.shape) + "\n" + "Label: " + str(label) + \
                    "\n******************************************"
        self.__logger.info(str(info_text), "ImagePlusLabelVisualizer:logImageAndLabelInformation")

    def __visualizeImageAndLabelWithBreak(self, image, label):
        """
        Visualizes the image and label given.
        Stops the program until some input is given.
        :param images: (np.array) The image to visualize.
        :param labels: (np.array) The label to visualize.
        """
        converterd_image = self.__checkAndConvertImage(image)
        converted_label = self.__checkAndConvertLabel(label)

        converterd_image = cv2.resize(converterd_image, (320, 320))

        cv2.imshow(str(converted_label), converterd_image)
        cv2.waitKey()

    def __checkAndConvertImage(self, image):
        """
        Checks if the given image is in right shape to be visualized. If not the shape will be converted properly.
        :param images: (np.array) The image to check.
        """
        img_shape = image.shape
        if(img_shape[0]<=3):
            image = image.transpose((1,2,0))
            image = image.astype(np.uint8)
            if img_shape[2] is 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

    def __checkAndConvertLabel(self, label):
        """
        Checks if the given label is in right shape to be visualized. If not the shape will be converted properly.
        :param label: (np.array) The label to check.
        """
        return label