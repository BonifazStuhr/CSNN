import tensorflow as tf

"""
Defines decay functions for learning rates and coefficients.
"""

def decreaseStdCoeff(coeff, global_step, max_steps):
    """
    Decreases the given coeff by: coeff*exp(2coeff * global_step/max_step). Used ti decrease the
    :param coeff: (Float) The coeff do decrease.
    :param global_step: (Float) The current global step.
    :param max_steps: (Float) The maximum steps to train.
    :return: decreased_coeff: (Float) The decreased coeff.
    """
    return tf.multiply(coeff, tf.exp(
        tf.negative(tf.multiply(tf.multiply(2.0, coeff), tf.truediv(global_step, max_steps)))))

def decreaseCoeff(coeff_start, coeff_end, global_step, max_steps):
    """
    Decreases the given coeff by: coeff_start*(coeff_end/coeffstart)^(global_step/max_step).
    Often used to decrease the neighborhood coeff in the standard SOM formulation.
    :param coeff_start: (Integer) The starting value of the coeff.
    :param global_step: (Float) The current global step.
    :param max_steps: (Float) The maximum steps to train.
    :return: decreased_coeff: (Float) The decreased coeff.
    """
    return tf.multiply(float(coeff_start), tf.pow(float(coeff_end)/float(coeff_start),
                                                  tf.truediv(global_step, float(max_steps))))
