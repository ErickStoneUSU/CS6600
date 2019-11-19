import cv2
import tflearn
import tensorflow as tf
from scipy.io import wavfile
from tflearn import fully_connected, dropout, input_data, conv_2d, max_pool_2d, regression


class Builder:
    # flat input 22500 grey scale values
    # simple 256 hidden layer, learning rate of 0.1, sigmoid
    # out 1,0 bee 0,1 no bee
    def image_components(self):
        net = input_data(shape=[None, 22500])
        net = fully_connected(net, 256, activation='sigmoid')
        net = fully_connected(net, 2, activation='sigmoid')
        return net

    def images_build(self):
        network = regression(self.image_components(), optimizer='sgd', metric='R2', loss='mean_square', learning_rate=0.1)
        model = tflearn.DNN(network)
        return model

    # input resized images to 150,150 and set to grey scale
    # 2 conv layers of 128 x 2, 2 256 hidden layers
    # out bee, cricket, noise
    def components_4(self):
        net = input_data(shape=[None, 150, 150, 1])
        net = conv_2d(net, 128, 2, activation='relu', regularizer="L2")
        net = max_pool_2d(net, 2)
        net = conv_2d(net, 128, 2, activation='relu', regularizer="L2")
        net = max_pool_2d(net, 2)
        net = fully_connected(net, 256, activation='sigmoid')
        net = fully_connected(net, 256, activation='sigmoid')
        net = fully_connected(net, 2, activation='sigmoid')
        return net

    def images_conv_build(self):
        network = regression(self.image_components(), metric='R2', loss='mean_square', learning_rate=1)
        model = tflearn.DNN(network)
        return model

    # dim is 44100, though others were tried
    # here we have a neural network with drop out
    # 2 hidden layers 64, 8196
    # out bee, cricket, noise
    def components_6(self, dim):
        net = input_data(shape=[None, dim])
        net = fully_connected(net, 64, activation='sigmoid')
        net = dropout(net, 0.5)
        net = fully_connected(net, 8196, activation='sigmoid')
        net = fully_connected(net, 3, activation='sigmoid')
        return net

    def audio_build(self, dim):
        network = regression(self.components_6(dim), optimizer='sgd', metric='R2', loss='mean_square', learning_rate=.01)
        model = tflearn.DNN(network)
        return model

    def components_7(self, dim):
        net = input_data(shape=[None, 64, 441, 100])
        net = conv_2d(net, 128, 2, activation='relu', regularizer="L2")
        net = max_pool_2d(net, 2)
        net = conv_2d(net, 128, 2, activation='relu', regularizer="L2")
        net = max_pool_2d(net, 2)
        net = fully_connected(net, 256, activation='sigmoid')
        net = fully_connected(net, 256, activation='sigmoid')
        net = fully_connected(net, 3, activation='sigmoid')
        return net

    def build_7(self, dim):
        network = regression(self.components_7(dim), optimizer='sgd', metric='R2', loss='mean_square', learning_rate=.01)
        model = tflearn.DNN(network)
        return model


def manip_audio(sam, aud):
    aud = aud[5000:44100 + 5000].reshape(1, 44100)
    return sam, aud


def manip_image(ann, im):
    # preprocess image
    # get input layer dims
    # resize image to input layer dims
    # initialize/edit any bad values
    grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) / 255
    if ann.inputs[0].shape.dims[1].value == 22500:
        grey.resize((1, 22500), refcheck=False)
    else:
        grey.resize((150, 150), refcheck=False)
    return grey


def fit_image_ann(ann, image_path):
    im = cv2.imread(image_path)
    im = manip_image(ann, im)
    return ann.predict(im)


def fit_audio_ann(ann, audio_path):
    # preprocess audio
    # trim ends
    sample, audio = wavfile.read(audio_path)
    sample, audio = manip_audio(sample, audio)
    return ann.predict(audio)


def load(ann_file_path):
    model = None
    if 'conv_images' in ann_file_path:
        model = Builder().images_conv_build()
    elif 'ann_images' in ann_file_path:
        model = Builder().images_build()
    if 'conv_audio' in ann_file_path:
        model = Builder().build_7(44100)
    elif 'buzz_ann' in ann_file_path:
        model = Builder().audio_build(44100)

    model.load(ann_file_path, True)
    return model


ann = load('conv_images')
print(fit_image_ann(ann, 'C:\\Users\\erick\\OneDrive\\Desktop\\CS6600\\Sources\\proj1\\data\\two_super\\testing\\bee\\0\\9.png'))

# ann = load('ann_images')
# print(fit_image_ann(ann, 'C:\\Users\\erick\\OneDrive\\Desktop\\CS6600\\Sources\\proj1\\data\\two_super\\testing\\bee\\0\\9.png'))

# ann = load('buzz_ann')
# print(fit_audio_ann(ann, 'C:\\Projects\\CS6600\\Sources\\proj1\\data\\BUZZ1\\noise\\noise1_192_168_4_9-2017-07-24_16-00-01.wav'))
