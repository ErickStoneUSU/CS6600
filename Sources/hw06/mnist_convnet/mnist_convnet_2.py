#!/usr/bin/python

################################################
# module: mnist_convnet_2.py
# bugs to vladimir dot kulyukin at usu dot edu
################################################

# uncomment if in Py2.
# from __future__ import division, print_function

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
import pickle
import tflearn.datasets.mnist as mnist

# X is the training data set; Y is the labels for X.
# testX is the testing data set; testY is the labels for testX.
X, Y, testX, testY = mnist.load_data(one_hot=True)
X, Y = shuffle(X, Y)
testX, testY = shuffle(testX, testY)
trainX = X[0:50000]
trainY = Y[0:50000]
validX = X[50000:]
validY = Y[50000:]
# make sure you reshape the training and testing
# data as follows.
trainX = trainX.reshape([-1, 28, 28, 1])
testX  = testX.reshape([-1, 28, 28, 1])

def save(obj, file_name):
    with open(file_name, 'wb') as fp:
        pickle.dump(obj, fp)

# change these two paths accordingly.
VALID_X_PATH = '/home/vladimir/research/mnist_convnets/valid_x.pck'
VALID_Y_PATH = '/home/vladimir/research/mnist_convnets/valid_y.pck'
save(validX, VALID_X_PATH)
save(validY, VALID_Y_PATH)

def build_tflearn_convnet_2(): 
    # your code here.

# the model is trained for a specified NUM_EPOCHS
# with a specified batch size; of course, you'll want
# to raise the number of epochs to some larger number.
NUM_EPOCHS = 30
BATCH_SIZE = 10
MODEL = build_tflearn_convnet_2()
MODEL.fit(trainX, trainY, n_epoch=NUM_EPOCHS,
          shuffle=True,
          validation_set=(testX, testY),
          show_metric=True,
          batch_size=BATCH_SIZE,
          run_id='MNIST_ConvNet_2')

# let me raise my text size at you: DO NOT FORGET TO PERSIST
# YOUR TRAINED MODELS. THIS IS AN IMPORTANT COMMAND.
SAVE_CONVNET_PATH = '/home/vladimir/research/mnist_convnets/hw06_my_net_02.tfl'
MODEL.save(SAVE_CONVNET_PATH)

# this is just to make sure that you've trained everything
# correctly.
#print(model.predict(testX[0].reshape([-1, 28, 28, 1])))

