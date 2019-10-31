import tflearn
from tflearn import input_data, fully_connected, regression


def components_1():
    net = input_data(shape=[None, 1024])
    net = fully_connected(net, 64, activation='sigmoid')
    net = fully_connected(net, 64, activation='sigmoid')
    net = fully_connected(net, 1, activation='sigmoid')
    return net


def build_1():
    network = regression(components_1(), optimizer='sgd', metric='R2', loss='mean_square', learning_rate=0.5)
    model = tflearn.DNN(network)
    return model


def components_2(dim):
    net = input_data(shape=[None, dim])
    net = fully_connected(net, 256, activation='sigmoid')
    net = fully_connected(net, 256, activation='sigmoid')
    net = fully_connected(net, 1, activation='sigmoid')
    return net


def build_2(dim):
    network = regression(components_2(dim), optimizer='sgd', metric='R2', loss='mean_square', learning_rate=0.1)
    model = tflearn.DNN(network)
    return model