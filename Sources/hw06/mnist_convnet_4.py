#!/usr/bin/python

################################################
# module: mnist_convnet_4.py
# bugs to vladimir dot kulyukin at usu dot edu
################################################

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression


def components_4():
    input_layer = input_data(shape=[None, 28, 28, 1])
    conv_layer_1 = conv_2d(input_layer, nb_filter=20, filter_size=3, activation='relu', name='conv_layer_1')
    pool_layer_1 = max_pool_2d(conv_layer_1, 2, name='pool_layer_1')
    conv_layer_2 = conv_2d(pool_layer_1, nb_filter=40, filter_size=3, activation='relu', name='conv_layer_2')
    pool_layer_2 = max_pool_2d(conv_layer_2, 2, name='pool_layer_2')
    conv_layer_3 = conv_2d(pool_layer_2, nb_filter=80, filter_size=3, activation='relu', name='conv_layer_3')
    pool_layer_3 = max_pool_2d(conv_layer_3, 2, name='pool_layer_3')
    conv_layer_4 = conv_2d(pool_layer_3, nb_filter=160, filter_size=3, activation='relu', name='conv_layer_4')
    pool_layer_4 = max_pool_2d(conv_layer_4, 2, name='pool_layer_4')
    fc_layer_1 = fully_connected(pool_layer_4, 100, activation='relu', name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 10, activation='softmax', name='fc_layer_2')
    return fc_layer_2


def build_tflearn_convnet_4():
    network = regression(components_4(), optimizer='sgd', loss='categorical_crossentropy', learning_rate=0.01)
    model = tflearn.DNN(network)
    return model
