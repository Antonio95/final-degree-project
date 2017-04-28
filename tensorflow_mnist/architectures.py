
"""

    Author: Antonio Mejias Gil (anmegi.95@gmail.com)
    Date: Early 2017
    Description:
        This file contains the core code to define some convnet architectures
        in the TensorFlow graph. The file tuner.py acts as a wrapper on it,
        initialises the rest of the graph and actually the networks.

"""

# Third-party modules
import tensorflow as tf
import tensorflow.contrib.slim as slim


# LeNet
#
# Input:                                        28x28x1
# CONV [n = 6, lxl = 5x5, s = 1; None]:         28x28x6
# POOL [max, lxl = 2x2, s = 2]:                 14x14x6
# CONV [n = 16, lxl = 5x5, s = 1] (no padding): 10x10x16
# POOL [max, lxl = 2x2, s = 2]:                 5x5x16
#  FC (relu):                                   120
#  FC (none):                                   84
#  FC (none):                                   10
def lenet(image, keep_prob, regularisation=0.001):

    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_regularizer=slim.l2_regularizer(regularisation)):
        l2 = slim.conv2d(image, 6, [5, 5], activation_fn=None, scope='conv_1')
        l3 = slim.max_pool2d(l2, [2, 2], scope='maxpool_1')
        l4 = slim.conv2d(l3, 16, [5, 5], padding='VALID', activation_fn=None, scope='conv_2')
        l5 = slim.max_pool2d(l4, [2, 2], scope='maxpool_2')
        l6 = slim.fully_connected(slim.flatten(l5), 120, scope='fc_1')
        l7 = slim.fully_connected(l6, 84, activation_fn=None, scope='fc_2')
        output = slim.fully_connected(l7, 10, activation_fn=None, scope='fc_3')

    return output


# TFNet
#
# Input: 28x28x1
# CONV [n = 32, lxl = 5x5, s = 1]:  28x28x32
# RELU:                             28x28x32
# POOL [max, lxl = 2x2, s = 2]:     14x14x32
# CONV [n = 64, lxl = 5x5, s = 1]:  14x14x64
# RELU:                             14x14x64
# POOL [max, lxl = 2x2, s = 2]:     7x7x64
# FC (relu):                        1024
# FC (none):                        10
def tfnet(image, keep_prob, regularisation=0.001):


    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_regularizer=slim.l2_regularizer(regularisation)):
        l2 = slim.conv2d(image, 32, [5, 5], scope='conv_1')
        l3 = slim.max_pool2d(l2, [2, 2], scope='maxpool_1')
        l4 = slim.conv2d(l3, 64, [5, 5], scope='conv_2')
        l5 = slim.max_pool2d(l4, [2, 2], scope='maxpool_2')
        l6 = slim.fully_connected(slim.flatten(l5), 1024, scope='fc_1')
        do = slim.dropout(l6, keep_prob, scope='dropout')
        output = slim.fully_connected(do, 10, activation_fn=None, scope='fc_2')

    return output


# VishNet
#
# Input:                                            28x28x1
# CONV [n = 64, lxl = 5x5, s = 1; ReLU]:            28x28x64
# POOL [max, lxl = 2x2, s = 2]:                     14x14x64
# CONV [n = 128, lxl = 5x5, s = 1; ReLU] (padding): 14x14x128
# POOL [max, lxl = 2x2, s = 2]:                     7x7x128
# CONV [n = 256, lxl = 5x5, s = 1; ReLU] (padding): 7x7x256
# POOL [max, lxl = 2x2, s = 2]:                     3x3x256
# FC (none):                                        10
def vishnet(image, keep_prob, regularisation=0.001):

    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_regularizer=slim.l2_regularizer(regularisation)):
        l2 = slim.conv2d(image, 64, [5, 5], scope='conv_1')
        l3 = slim.max_pool2d(l2, [2, 2], scope='maxpool_1')
        l4 = slim.conv2d(l3, 128, [5, 5], scope='conv_2')
        l5 = slim.max_pool2d(l4, [2, 2], scope='maxpool_2')
        l6 = slim.conv2d(l5, 256, [5, 5], scope='conv_3')
        l7 = slim.max_pool2d(l6, [2, 2], stride=1, scope='maxpool_3')
        do = slim.dropout(slim.flatten(l7), keep_prob, scope='dropout')
        output = slim.fully_connected(do, 10, scope='fc')

    return output
