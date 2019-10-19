import tflearn
from tflearn import input_data, fully_connected, regression


def components_1():
    input_layer = input_data(shape=[None, 32, 32, 3])
    fc_layer_1 = fully_connected(input_layer, 32, activation='relu', name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 32, activation='softmax', name='fc_layer_2')
    fc_layer_3 = fully_connected(fc_layer_2, 3, activation='softmax', name='fc_layer_3')
    return fc_layer_3


def build_1():
    network = regression(components_1(), optimizer='sgd', loss='categorical_crossentropy', learning_rate=0.01)
    model = tflearn.DNN(network)
    return model