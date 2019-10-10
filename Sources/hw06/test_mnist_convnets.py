#pragma once

################################################
# module: mnist_convnet_load.py
# bugs to vladimir dot kulyukin at usu dot edu
################################################

import pickle
import time

import numpy as np
import tensorflow
import tflearn
import tflearn.datasets.mnist as mnist
from scipy import stats
from tflearn.data_utils import shuffle

from hw06.mnist_convnet_1 import components_1, build_tflearn_convnet_1
from hw06.mnist_convnet_2 import components_2, build_tflearn_convnet_2
from hw06.mnist_convnet_3 import components_3, build_tflearn_convnet_3
from hw06.mnist_convnet_4 import components_4, build_tflearn_convnet_4
from hw06.mnist_convnet_5 import components_5, build_tflearn_convnet_5


def load_mnist_convnet_1(path): return get_model(components_1, path)


def load_mnist_convnet_2(path): return get_model(components_2, path)


def load_mnist_convnet_3(path): return get_model(components_3, path)


def load_mnist_convnet_4(path): return get_model(components_4, path)


def load_mnist_convnet_5(path): return get_model(components_5, path)


def get_model(components, path):
    model = tflearn.DNN(components())
    model.load(path)
    return model


def load(file_name):
    with open(file_name, 'rb') as fp:
        obj = pickle.load(fp)
    return obj


def save(obj, file_name):
    with open(file_name, 'wb') as fp:
        pickle.dump(obj, fp)


def test_tflearn_convnet_model(convnet_model, validX, validY):
    results = []
    for i in range(len(validX)):
        prediction = convnet_model.predict(validX[i].reshape([-1, 28, 28, 1]))
        results.append(np.argmax(prediction, axis=1)[0] == np.argmax(validY[i]))
    return sum((np.array(results) == True)) / len(results)


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


def get_data():
    x, y, test_x, test_y = mnist.load_data(one_hot=True)
    x, y = shuffle(x, y)
    test_x, test_y = shuffle(test_x, test_y)
    train_x = x[0:50000]
    train_y = y[0:50000]
    valid_x = x[50000:]
    valid_y = y[50000:]
    # make sure you reshape the training and testing
    # data as follows.
    train_x = train_x.reshape([-1, 28, 28, 1])
    test_x = test_x.reshape([-1, 28, 28, 1])
    return train_x, train_y, test_x, test_y, valid_x, valid_y


def start_up_model(build, run_id, num):
    start_time = time.time()
    with tensorflow.Graph().as_default():
        print('starting ' + run_id)
        NUM_EPOCHS = 2
        BATCH_SIZE = 10
        MODEL = build()
        MODEL.fit(train_x, train_y, n_epoch=NUM_EPOCHS,
                  shuffle=True,
                  validation_set=(test_x, test_y),
                  show_metric=True,
                  batch_size=BATCH_SIZE,
                  run_id=run_id)
        MODEL.save('/data/hw06_my_net_0' + num + '.tfl')
        print('Desired y: ' + str(test_y[0]))
        print(MODEL.predict(test_x[0].reshape([-1, 28, 28, 1])))
    end_time = time.time()
    return round(end_time - start_time, 2)


def test_20(valid_x, valid_y, model, full=False):
    tests = []
    num_tests = 20
    for j in range(num_tests):
        i = np.random.randint(0, len(valid_x) - 1)
        prediction = model.predict(valid_x[i].reshape([-1, 28, 28, 1]))
        if full:
            print(np.argmax(prediction, axis=1)[0] == np.argmax(valid_y[i]))
        tests.append(np.argmax(prediction, axis=1)[0] == np.argmax(valid_y[i]))
    print(sum((np.array(tests) == True)) / num_tests)


def test_model(tfl, loader):
    with tensorflow.Graph().as_default():
        model = loader(tfl)
        #test_20(valid_x, valid_y, model, True)
        print('ConvNet accuracy = {}'.format(test_tflearn_convnet_model(model, valid_x, valid_y)))


train_x, train_y, test_x, test_y, valid_x, valid_y = get_data()

times = []
times.append(start_up_model(build=build_tflearn_convnet_1, run_id='MNIST_ConvNet_1', num='1'))
times.append(start_up_model(build=build_tflearn_convnet_2, run_id='MNIST_ConvNet_2', num='2'))
times.append(start_up_model(build=build_tflearn_convnet_3, run_id='MNIST_ConvNet_3', num='3'))
times.append(start_up_model(build=build_tflearn_convnet_4, run_id='MNIST_ConvNet_4', num='4'))
times.append(start_up_model(build=build_tflearn_convnet_5, run_id='MNIST_ConvNet_5', num='5'))
print(times)
test_model('/data/hw06_my_net_01.tfl', load_mnist_convnet_1)
test_model('/data/hw06_my_net_02.tfl', load_mnist_convnet_2)
test_model('/data/hw06_my_net_03.tfl', load_mnist_convnet_3)
test_model('/data/hw06_my_net_04.tfl', load_mnist_convnet_4)
test_model('/data/hw06_my_net_05.tfl', load_mnist_convnet_5)
