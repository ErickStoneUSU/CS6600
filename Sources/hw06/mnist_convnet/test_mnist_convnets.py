
## uncomment this import if you use Py2.
## from __future__ import division, print_function

################################################
# module: mnist_convnet_load.py
# bugs to vladimir dot kulyukin at usu dot edu
################################################

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
import tflearn.datasets.mnist as mnist
import numpy as np
import pickle
from scipy import stats

# uncomment and change these paths accordingly
net_path_1 = '/home/vladimir/research/mnist_convnets/hw06_my_net_01.tfl'
#net_path_2 = '/home/vladimir/research/mnist_convnets/hw06_my_net_02.tfl'
#net_path_3 = '/home/vladimir/research/mnist_convnets/hw06_my_net_03.tfl'
#net_path_4 = '/home/vladimir/research/mnist_convnets/hw06_my_net_04.tfl'
#net_path_5 = '/home/vladimir/research/mnist_convnets/hw06_my_net_05.tfl'

def load_mnist_convnet_1(path):
    input_layer = input_data(shape=[None, 28, 28, 1])
    conv_layer_1  = conv_2d(input_layer,
                            nb_filter=20,
                            filter_size=5,
                            activation='relu',
                            name='conv_layer_1')
    pool_layer_1  = max_pool_2d(conv_layer_1, 2,
                                name='pool_layer_1')
    conv_layer_2 = conv_2d(pool_layer_1,
                           nb_filter=40,
                           filter_size=3,
                           activation='relu',
                           name='conv_layer_2')
    pool_layer_2 = max_pool_2d(conv_layer_2, 2,
                               name='pool_layer_2')
    fc_layer_1  = fully_connected(pool_layer_2, 100,
                                  activation='relu',
                                  name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 10,
                                 activation='softmax',
                                 name='fc_layer_2')
    model = tflearn.DNN(fc_layer_2)
    # load the trained and persisted network.
    model.load(path)
    return model

def load_mnist_convnet_2(path):
    # your code here.

def load_mnist_convnet_3(path):
    # your code here.

def load_mnist_convnet_4(path):
    # your code here.

def load_mnist_convnet_5(path):
    # your code

net_model_1 = load_mnist_convnet_1(net_path_1)
#net_model_2 = load_mnist_convnet_2(net_path_2)
#net_model_3 = load_mnist_convnet_3(net_path_3)
#net_model_4 = load_mnist_convnet_4(net_path_4)
#net_model_5 = load_mnist_convnet_5(net_path_5)

def load(file_name):
    with open(file_name, 'rb') as fp:
        obj = pickle.load(fp)
    return obj

# load the validation data; change these directories accordingly
# change these two paths accordingly.
VALID_X_PATH = '/home/vladimir/research/mnist_convnets/valid_x.pck'
VALID_Y_PATH = '/home/vladimir/research/mnist_convnets/valid_y.pck'
validX = load(VALID_X_PATH)
validY = load(VALID_Y_PATH)

def test_tflearn_convnet_model(convnet_model, validX, validY):
    results = []
    for i in range(len(validX)):
        prediction = convnet_model.predict(validX[i].reshape([-1, 28, 28, 1]))
        results.append(np.argmax(prediction, axis=1)[0] == \
                       np.argmax(validY[i]))
    return sum((np.array(results) == True))/len(results)

def run_convnet_ensemble(net_ensemble, sample):
    test_results = []
    for net in net_ensemble:
        pred = net.predict(sample.reshape([-1, 28, 28, 1]))
        test_results.append(np.argmax(pred, axis=1)[0])
    return stats.mode(test_results).mode[0]

def evaluate_convnet_ensemble(net_ensemble, validX, validY):
    assert len(validX) == len(validY)
    test_results = []
    for x, y in zip(validX, validY):
        test_results.append(run_convnet_ensemble(net_ensemble, x), y)
    return sum(int(x == y) for x, y in test_results), len(validX)

if __name__ == '__main__':
    print('ConvNet 1 accuracy = {}'.format(test_tflearn_convnet_model(net_model_1, validX, validY)))

# let's test the persisted network on 20
# randomly selected validation samples.
#tests = []
#num_tests = 20
#for j in xrange(num_tests):
#    i = np.random.randint(0, len(validX)-1)
#    prediction = model.predict(validX[i].reshape([-1, 28, 28, 1]))
#
    # print(np.argmax(prediction, axis=1)[0] == np.argmax(validY[i]))
#    tests.append(np.argmax(prediction, axis=1)[0] == np.argmax(validY[i]))
#
#print(sum((np.array(tests) == True))/num_tests)
