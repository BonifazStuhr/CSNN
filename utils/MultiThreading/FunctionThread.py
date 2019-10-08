import threading

class FunctionThread(threading.Thread):
   """
   The FunctionThread runs a function with the given *args in a separate thread.

   :Attributes:
      __function: (Function) The function to execute threaded.
      __args:     (*args) The *args of the function.
   """

   def __init__(self, function, *args):
      """
      Constructor, initialize member variables and the thread.
      :param function: (Function) The function to execute threaded.
      :param *args: (*args) The *args of the function.
      """
      self.__function = function
      self.__args = args
      threading.Thread.__init__(self)

   def run(self):
      """
      Executes the given function with the given *args.
      """
      self.__function(*self.__args)

