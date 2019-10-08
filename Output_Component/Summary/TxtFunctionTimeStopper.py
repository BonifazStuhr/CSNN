import os
import time
import platform
from tensorflow.python.client import device_lib


class TxtFunctionTimeStopper():
    """
    The TxtFunctionTimeStopper writes time information of given functions into a specified text file.
    Starting two functions with same name is not supported jet!

     :Attributes:
        __writer: (file_object) The summary writer for the time information.
        __runningFunctions: (Dictionary) For each running function (key) the time information (value).
    """
    def __init__(self, dir):
        """
        Constructor, initialize member variables.
        :paramdir: (String) The path to the directory, where the summary will be saved under /logs.
        """
        if not os.path.exists(dir + "/txtlogs/"):
            os.makedirs(dir + "/txtlogs/")
            self.__writer = open(dir + "/txtlogs/functionTimes.txt", "a")
            self.__writer.write("Specs: \n "+
                            "Processor: " + platform.processor() + "\n" +
                            "Platform: " + platform.platform() + "\n" +
                            "GPUs: " + str(device_lib.list_local_devices()) + "\n")
        else:
            self.__writer = open(dir + "/txtlogs/functionTimes.txt", "a")
        self.__runningFunctions = {}

    def startNewFunction(self, func):
        """
        Start the stopwatch for the given function.
        :param func: (String) The function for witch the timer is started.
        """
        self.__runningFunctions[func] = time.time()

    def stopFunctionAndWrite(self, func, comment=""):
        """
        Ends the stopwatch for the given function and writes time information and the given comment in the specified
        text file.
        :param func: (String) The function to stop.
        :param comment: (String) Additional information to write.
        """
        end_time = time.time()
        start_time = self.__runningFunctions.pop(func, None)
        self.__writer.write("Function: " + func + " | Execution Time: " + str(end_time-start_time) +
                    " | Comment: " + comment + " | Start time: " + str(start_time) + " | End time: " + str(end_time))

    def __exit__(self, *args):
        """
        To properly close the class when leaving 'with' statements.
        ::param *args
        """
        self.__writer.close()


