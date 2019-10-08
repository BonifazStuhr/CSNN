#Copyright (c) <year> <copyright holders>

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

"""
Entry File, which contains the main method.
"""
import os
import time

from Controller_Component.Controller import Controller


def main():
    """
    Main method which initialises and starts the execution via the controller.
    The type of the execution specified in the controllerConfig.
    """
    config_path = os.path.dirname(os.path.realpath(__file__)) + "/controllerConfig.json"
    controller = Controller(config_path)

    print("Main: Starting initialisation ...")
    start_initialisation_time = time.time()
    initialisation_ok = controller.init()
    end_initialisation_time = time.time()
    print("#########FINSIHED INITIALISATION##########")
    print("Initialisation successful: ", initialisation_ok)
    print("Time for initialisation: ", end_initialisation_time-start_initialisation_time, "s")
    print("##########################################")

    print("Main: Starting execution ...")
    start_execution_time = time.time()
    execution_ok = controller.execute()
    end_execution_time = time.time()
    print("############FINSIHED EXECUTION############")
    print("Execution successful: ", execution_ok)
    print("Time for execution: ", end_execution_time-start_execution_time, "s")
    print("##########################################")

main()
