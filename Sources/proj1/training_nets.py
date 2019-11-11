import glob
import random

import cv2
import numpy as np
import tflearn
from numpy import int16, float16, float32
from scipy.io import wavfile
from tflearn import input_data, fully_connected, regression, conv_2d, max_pool_2d, dropout


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
        net = input_data(shape=[None, 64, 441, 100], dtype=float32)
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




class Loader:
    def get_all_im(self):
        x = []
        y = []
        l = 150
        dim = l * l
        d = (int(l / 2), int(l / 2))
        # get the rotation matrices
        M = cv2.getRotationMatrix2D(d, 90, 1.0)
        M2 = cv2.getRotationMatrix2D(d, 180, 1.0)
        for f in glob.glob('data/**/**valid/**/*.png'):
            f = f.replace('\\', '/')
            im = cv2.imread(f)
            grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) / 255
            grey.resize((l, l), refcheck=False)
            # NAN check
            if grey.all() != grey.all():
                print('bad')
            else:
                # append date
                x.append(grey.reshape(1, dim))
                if 'no' in f:
                    y.append([0, 1])
                else:
                    y.append([1, 0])

                # append 90 rotation
                grey2 = cv2.warpAffine(grey, M, (l, l))
                x.append(grey2.reshape(1, dim))
                if 'no' in f:
                    y.append([0, 1])
                else:
                    y.append([1, 0])

                # append 180 rotation
                grey3 = cv2.warpAffine(grey, M2, (l, l))
                x.append(grey3.reshape(1, dim))
                if 'no' in f:
                    y.append([0, 1])
                else:
                    y.append([1, 0])

        e = list(zip(x, y))
        random.shuffle(e)
        x, y = zip(*e)
        return np.asarray(x).reshape(-1, dim), np.asarray(y).reshape(-1, 2)

    def get_all_audio(self, dim):
        x = []
        y = []
        for f in glob.glob('data/**/**/**/*.wav'):
            f = f.replace('\\', '/')
            sr, a = wavfile.read(f)
            # NAN check
            if a.all() != a.all():
                print('bad')
            else:
                # chop off the end to ignore the clicking of stopping
                # chop off beginning because it is weird
                # also to make them the same length
                x.append(a[5000:dim + 5000])
                if 'bee' in f:
                    y.append([1, 0, 0])
                elif 'cricket' in f:
                    y.append([0, 1, 0])
                elif 'noise' in f:
                    y.append([0, 0, 1])
                else:
                    y.append([0, 0, 0])

        e = list(zip(x, y))
        random.shuffle(e)
        x, y = zip(*e)
        return np.asarray(x, dtype=float32), np.asarray(y).reshape(-1, 3)


class Trainer:
    l = Loader()
    b = Builder()

    def image_train(self):
        x, y = self.l.get_all_im()
        # 70% train, 10% test, 20% validate
        x_train = x[:int(len(x) * (0.7))]
        y_train = y[:int(len(x) * (0.7))]
        x_test = x[int(len(x) * (0.7))+1: int(len(x) * (0.8))]
        y_test = y[int(len(x) * (0.7))+1: int(len(x) * (0.8))]
        x_valid = x[int(len(x) * (0.8))+1:]
        y_valid = x[int(len(x) * (0.8))+1:]

        model = self.b.images_build()
        model.fit(x_train, y_train, validation_set=(x_test, y_test), show_metric=True, n_epoch=1000)
        # print(model.evaluate(x_valid, y_valid))
        model.save('ann_images')

    def image_conv_train(self):
        x, y = self.l.get_all_im()
        # 70% train, 10% test, 20% validate
        x_train = x[:int(len(x) * (0.7))]
        y_train = y[:int(len(y) * (0.7))]
        x_test = x[int(len(x) * (0.7))+1: int(len(x) * (0.8))]
        y_test = y[int(len(y) * (0.7))+1: int(len(y) * (0.8))]
        x_valid = x[int(len(x) * (0.8))+1:]
        y_valid = y[int(len(y) * (0.8))+1:]

        model = self.b.images_build()
        # model.load('conv_images')
        model.fit(x_train, y_train, validation_set=(x_test, y_test), show_metric=True, n_epoch=1000)
        print(model.evaluate(x_valid, y_valid))
        model.save('conv_images')

    def audio_train(self):
        l = Loader()
        d = 44100
        x, y = l.get_all_audio(d)
        # 70% train, 10% test, 20% validate
        x_train = x[:int(len(x) * (0.7))]
        y_train = y[:int(len(y) * (0.7))]
        x_test = x[int(len(x) * (0.7))+1: int(len(x) * (0.8))]
        y_test = y[int(len(y) * (0.7))+1: int(len(y) * (0.8))]
        x_valid = x[int(len(x) * (0.8))+1:]
        y_valid = y[int(len(y) * (0.8))+1:]
        model = self.b.audio_build(d)
        model.load('buzz_ann')
        model.fit(x_train, y_train, validation_set=(x_test, y_test), show_metric=True, n_epoch=500)
        print(model.evaluate(x_valid, y_valid))
        model.save('buzz_ann')


    def audio_conv_train(self):
        l = Loader()
        d = 44100
        x, y = l.get_all_audio(d)
        x = x.reshape(-1, 441, 100)
        # 70% train, 10% test, 20% validate
        x_train = x[:int(len(x) * (0.7))]
        y_train = y[:int(len(y) * (0.7))]
        x_test = x[int(len(x) * (0.7))+1: int(len(x) * (0.8))]
        y_test = y[int(len(y) * (0.7))+1: int(len(y) * (0.8))]
        x_valid = x[int(len(x) * (0.8))+1:]
        y_valid = y[int(len(y) * (0.8))+1:]
        model = self.b.build_7(d)
        # model.load('buzz_ann')
        model.fit(x_train, y_train, validation_set=(x_test, y_test), show_metric=True, n_epoch=500)
        print(model.evaluate(x_valid, y_valid))
        model.save('buzz_conv_ann')


Trainer().image_conv_train()
