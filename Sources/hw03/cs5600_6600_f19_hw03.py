# /usr/bin/python

############################
# Module: cs5600_6600_f19_hw03.py
# Erick Stone
# A02217762
############################

from hw03.network import Network
from hw03.mnist_loader import load_data_wrapper
import random
import pickle
import numpy as np
import os
from scipy import stats

# load training, validation, and testing MNIST data
train_d, valid_d, test_d = load_data_wrapper()

# define your networks
net1 = Network([784, 30, 60, 120, 10])  # 5
net2 = Network([784, 23, 23, 23, 23, 10])  # 6
net3 = Network([784, 30, 10])  # 3
net4 = Network([784, 23, 23, 10])  # 4
net5 = Network([784, 100, 10])  # 3

# define an ensemble of 5 nets
networks = (net1, net2, net3, net4, net5)
eta_vals = (0.1, 0.25, 0.3, 0.4, 0.5)
mini_batch_sizes = (5, 10, 15, 20)

num_epochs = 5
path = 'pickles/high scoring/'


# save() function to save the trained network to a file
def save(ann, file_name):
    with open(file_name, 'wb') as fp:
        pickle.dump(ann, fp)


# restore() function to restore the file
def load(file_name):
    with open(file_name, 'rb') as fp:
        nn = pickle.load(fp)
    return nn


# train networks
def train_nets(networks, eta_vals, mini_batch_sizes, num_epochs, path):
    for n in networks:
        for m in mini_batch_sizes:
            for e in eta_vals:
                ne = n
                ne.SGD(train_d, num_epochs, m, e, test_data=test_d)
                # arch_eta * 100_minibatch.pck
                # record to pickle file
                save(ne, path + 'net_' + str(ne.sizes).replace(', ','_').replace('[','').replace(']','') +
                     '_' + str(int(e * 100)) + '_' + str(m) + '.pck')


def load_nets(path):
    # your code here
    nets = []
    for (_,_,files) in os.walk(path):
        break
    for f in files:
        nets.append(load(path + f))
    return tuple(nets)


# evaluate net ensemble.
def evaluate_net_ensemble(net_ensemble, test_data):
    # your code here
    res = []
    for (x, y) in test_data:
        ave = []
        for n in net_ensemble:
            ave.append(np.argmax(n.feedforward(x)))

        res.append(int(stats.mode(ave)[0] == y))
    return sum(res), len(res)


# train_nets(networks, eta_vals, mini_batch_sizes, num_epochs, path)
nets = load_nets(path)
print(evaluate_net_ensemble(nets, valid_d))

