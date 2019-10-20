import tflearn
from tflearn import input_data, fully_connected, regression, max_pool_2d, reshape


def components_1():
    input_layer = input_data(shape=[None, 1536, 2])
    # rs_layer = reshape(input_layer, new_shape=[96, 32], name='reshape_1')
    fc_layer_1 = fully_connected(input_layer, 256, activation='relu', name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 256, activation='relu', name='fc_layer_2')
    fc_layer_3 = fully_connected(fc_layer_2, 64, activation='relu', name='fc_layer_3')
    return fc_layer_3


def build_1():
    network = regression(components_1(), optimizer='sgd', loss='categorical_crossentropy', learning_rate=0.01)
    model = tflearn.DNN(network)
    return model