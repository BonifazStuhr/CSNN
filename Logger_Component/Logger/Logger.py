import datetime
import os
import cv2
import glob
import sys

from termcolor import colored

class LogLevel:
    """
    A "enum" class for the loglevel flags.
    """
    DEBUG = 1
    INFO = 2
    WARNING = 4
    ERROR = 8
    TRAIN = 16
    VAL = 32
    INFER = 64
    ALL = 127

    Color = {1: "blue", 2: "white", 4: "magenta", 8: "red", 16: "yellow", 32: "green", 64: "cyan"}
    String = {1: "DEBUG", 2: "INFO", 4: "WARNING", 8: "ERROR", 16: "TRAIN", 32: "VAL", 64: "INFER"}


class Logger:
    """
    Class for logging messages.
    Messages that correspond to the LogLevel are written to a logfile and output to the console.

    :Attributes:
        __name:          (String) The name of the logger and the corresponding logfile.
        __log_level:     (LogLevel) Specifies which messages are logged.
        __print_level:   (LogLevel) Specifies which messages are printed on the console.
        __file_name:     (String) Name of the logfile.
        __folder:        (String) Relative path to the logfile.
        __mode:          (Char) The write mode.
        __index_file:    (String) The connection to the logfile into a html file..
        __counter:       (Integer) For the naming of the pictures, so that no two pictures are written at the same time.
        __save_images:   (Boolean) Flag if images should be saved.
        __log_html:      (Boolean) If true, logs will be written in a html file.
    """

    def __init__(self, name, folder="", append=False, log_level=LogLevel.ALL, print_level=LogLevel.ALL, log_html=False):
        """
        Constructor for a Logger.
        :param name: (String) The name of the logger. Name of the subdirectory containing the logfiles.
        :param folder: (String) Relative path to the log folder.
        :param append: (Boolean) Determines whether log messages are attached to an existing logfile or whether the old
                        logfile is overwritten. Applies only to the first call of the logger. False by default.
        :param log_level: (LogLevel) Specifies which messages are logged into a html file.
        :param print_level: (LogLevel) Specifies which messages are printed on the console.
        :param log_html: (Boolean) If true, logs will be written in a html file.
        """
        self.__name = name
        self.__log_level = log_level
        self.__print_level = print_level
        self.__log_html = log_html
        self.__file_name = "index.html"
        self.__folder = name + "/"
        self.__mode = ""
        self.__counter = 0
        self.__save_images = True

        if folder != "":
            self.__folder = folder + "/" + name + "/"

            if not os.path.exists(folder):
                os.mkdir(folder)

        if not os.path.exists(self.__folder):
            os.mkdir(self.__folder)

        if not os.path.exists(self.__folder + "images"):
            os.mkdir(self.__folder + "images")

        if append:
            self.__mode = "a"
        else:
            self.__mode = "w"

            for image in glob.glob(self.__folder + "images/*.jpg"):
                os.remove(image)

        self.__index_file = open(self.__folder + "/" + self.__file_name, self.__mode)

    def __del__(self):
        """
        Destructor.
        Closes the link to the logfile.
        """
        self.__index_file.close()

    def getName(self):
        """
        Getter for the logger name.
        :return: (String) The name of the logger.
        """
        return self.__name

    def setLogLevel(self, level):
        """
        Setter for the loglevel.
        Determines which levels are written to the logfile.
        :param level : (LogLevel) The loglevel.
        :example:
            logger.setLogLevel(LogLevel.DEBUG)
            logger.setLogLevel(LogLevel.DEBUG | LogLevel.INFO)
            logger.setLogLevel(LogLevel.DEBUG | LogLevel.WARNING | LogLevel.ERROR)
            logger.setLogLevel(LogLevel.ALL)
        """
        self.__log_level = level

    def setLogHtml(self, log_html):
        """
        Setter for the log_html.
        Determines if logs ar written to a html file.
        :param log_html : (Boolean) If true, logs will be written in a html file.
        :example:
            logger.setLogHtml(True)
        """
        self.__log_html = log_html


    def setPrintLevel(self, level):
        """
        Setter for the print level.
        Determines which levels are output on the console.
        :param level : (LogLevel) The print level.
        :example:
            logger.setPrintLevel(LogLevel.DEBUG)
            logger.setPrintLevel(LogLevel.DEBUG | LogLevel.INFO)
            logger.setPrintLevel(LogLevel.DEBUG | LogLevel.WARNING | LogLevel.ERROR)
            logger.setPrintLevel(LogLevel.ALL)
        """
        self.__print_level = level

    def saveImages(self, value):
        """
        Determines whether images are saved (True) or not (False).
        :param value: (Boolean) Value for the __save_image flag.
        """
        self.__save_images = value

    def debug(self, message, sender=None, image=None):
        """
        Logs a debug message.
        :param message: (String) The message.
        :param sender: (String) The sender of the message. None by default.
        :param image: (Array) An image which is attached to the message in the logfile. None by default.
        """
        if ((LogLevel.DEBUG & self.__log_level) > 0) and self.__log_html:
            self.__log(LogLevel.DEBUG, message, sender, image)

        if (LogLevel.DEBUG & self.__log_level) > 0:
            self.__print(LogLevel.DEBUG, message, sender)

    def info(self, message, sender=None, image=None, overwrite=False):
        """
        Logs a info message.
        :param message: (String) The message.
        :param sender: (String) The sender of the message. [None] by default.
        :param image: (Array) An image which is attached to the message in the logfile. None by default.
        """
        if ((LogLevel.INFO & self.__log_level) > 0) and self.__log_html:
            self.__log(LogLevel.INFO, message, sender, image)

        if (LogLevel.INFO & self.__log_level) > 0:
            self.__print(LogLevel.INFO, message, sender, overwrite=overwrite)

    def warning(self, message, sender=None, image=None):
        """
         Logs a warning message.
        :param message: (String) The message.
        :param sender: (String) The sender of the message. [None] by default.
        :param image: (Array) An image which is attached to the message in the logfile. None by default.
        """
        if ((LogLevel.WARNING & self.__log_level) > 0) and self.__log_html:
            self.__log(LogLevel.WARNING, message, sender, image)

        if (LogLevel.WARNING & self.__log_level) > 0:
            self.__print(LogLevel.WARNING, message, sender)

    def error(self, message, sender=None, image=None):
        """
        Logs a error message.
        :param message: (String) The message.
        :param sender: (String) The sender of the message. [None] by default.
        :param image: (Array) An image which is attached to the message in the logfile. None by default.
        """
        if ((LogLevel.ERROR & self.__log_level)) and self.__log_html > 0:
            self.__log(LogLevel.ERROR, message, sender, image)

        if (LogLevel.ERROR & self.__log_level) > 0:
            self.__print(LogLevel.ERROR, message, sender)

    def train(self, message, sender=None, count=None, total=None, prefix='', suffix='', image=None):
        """
        Logs a train message.
        :param message: (String) The message.
        :param sender: (String) The sender of the message. None by default.
        :param count: (Integer) The current count. None by default.
        :param total : (Integer) The total count. None by default.
        :param prefix : (String) The suffix of the progressbar. '' by default.
        :param suffix : (String) The suffix of the progressbar. '' by default.
        :param image: (Array) An image which is attached to the message in the logfile. None by default.
        """
        if ((LogLevel.TRAIN & self.__log_level)) and self.__log_html > 0:
            self.__log(LogLevel.TRAIN, message, sender, image)

        if (LogLevel.TRAIN & self.__log_level) > 0:
            self.__print(LogLevel.TRAIN, message, sender)

        sys.stdout.flush()

        if total != None:
            self.__printProgress(count, total, LogLevel.TRAIN, prefix, suffix)

    def val(self, message, sender=None, count=None, total=None, prefix='', suffix='', image=None):
        """
        Logs a val message.
        :param message: (String) The message.
        :param sender: (String) The sender of the message. None by default.
        :param count: (Integer) The current count. None by default.
        :param total : (Integer) The total count. None by default.
        :param prefix : (String) The suffix of the progressbar. '' by default.
        :param suffix : (String) The suffix of the progressbar. '' by default.
        :param image: (Array) An image which is attached to the message in the logfile. None by default.
        """
        if ((LogLevel.VAL & self.__log_level)) and self.__log_html > 0:
            self.__log(LogLevel.VAL, message, sender, image)

        if (LogLevel.VAL & self.__log_level) > 0:
            self.__print(LogLevel.VAL, message, sender)

        if total != None:
            self.__printProgress(count, total, LogLevel.VAL, prefix, suffix)

    def infer(self, message, sender=None, count=None, total=None, prefix='', suffix='', image=None):
        """
        Logs a infer message.
        :param message: (String) The message.
        :param sender: (String) The sender of the message. None by default.
        :param count: (Integer) The current count. None by default.
        :param total : (Integer) The total count. None by default.
        :param prefix : (String) The suffix of the progressbar. '' by default.
        :param suffix : (String) The suffix of the progressbar. '' by default.
        :param image: (Array) An image which is attached to the message in the logfile. None by default.
        """
        if ((LogLevel.INFER & self.__log_level)) and self.__log_html > 0:
            self.__log(LogLevel.INFER, message, sender, image)

        if (LogLevel.INFER & self.__log_level) > 0:
            self.__print(LogLevel.INFER, message, sender)

        if total!=None:
            self.__printProgress(count, total, LogLevel.INFER, prefix, suffix)

    def __log(self, level, message, sender, image):
        """
        Write a message in the logfile.
        :param level: (LogLevel) The loglevel of the message.
        :param message: (String) The message.
        :param sender: (String) The sender of the message. None by default.
        :param image: (Array) An image which is attached to the message in the logfile. None by default.
        """
        if sender is None:
            sender = "UNDEFINED"

        output = "<font style='color: " + LogLevel.Color[
            level] + "; font-family: courier new; font-weight: bold'>" + datetime.datetime.now().strftime(
            "%Y-%m-%d_%H:%M:%S") + ": " + LogLevel.String[
                     level] + " from " + sender + ": '" + message + "'" + "</font><br>"

        if image is not None and self.__save_images:
            img = "images/" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f") + "_" + str(
                self.__counter) + ".jpg"
            self.__counter += 1
            cv2.imwrite(self.__folder + img, image)
            output += "<a href='" + img + "'><img style='max-width:300px;' src='" + img + "' /></a><br>"

        self.__index_file.write(output + "<br>")

    def __print(self, level, message, sender, overwrite=False):
        """
        Write a message to the console.
        :param level: (LogLevel) The loglevel of the message.
        :param message: (String) The message.
        :param sender: (String) The sender of the message. None by default.
        """
        if sender is None:
            sender = "UNDEFINED"

        end = '\n'
        if overwrite:
            end = '\r'

        print(colored(self.__name + ": " + datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + ": " + LogLevel.String[
            level] + " from " + sender + ": '" + message + "'", LogLevel.Color[level]), end=end, flush=True)

    def __printProgress(self, count, total, log_level, prefix='', suffix='', decimals=2):
        """
        Prints a progressbar to the console.
        :param count: (Integer) The current count.
        :param total: (Integer) The total count.
        :param log_level: (LogLevel) The loglevel of the progressbar.
        :param prefix: (String) The suffix of the progressbar. '' by default.
        :param suffix: (String) The suffix of the progressbar. '' by default.
        :param decimals: (Integer) Number of decimals in percent complete.
        """
        fill='█'
        length = 100
        percent = ("{0:." + str(decimals) + "f}").format(100 * (count / float(total)))
        filledLength = int(length * count // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(colored('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), LogLevel.Color[log_level]), end='\r')
        # Print New Line on Complete
        if count == total:
            print()