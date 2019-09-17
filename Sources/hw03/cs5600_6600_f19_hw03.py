#/usr/bin/python

############################
# Module: cs5600_6600_f19_hw03.py
# Your name
# Your A#
############################

from network import Network
from mnist_loader import load_data_wrapper
import random
import pickle as cPickle
import numpy as np

# load training, validation, and testing MNIST data
train_d, valid_d, test_d = load_data_wrapper()

# define your networks
net1 = None
net2 = None
net3 = None
net4 = None
net5 = None

# define an ensemble of 5 nets
networks = (net1, net2, net3, net4, net5)
eta_vals = (0.1, 0.25, 0.3, 0.4, 0.5)
mini_batch_sizes = (5, 10, 15, 20)

# train networks
def train_nets(networks, eta_vals, mini_batch_sizes, num_epochs, path):
    # your code here
    pass

def load_nets(path):
    # your code here
    pass

# evaluate net ensemble.
def evaluate_net_ensemble(net_ensemble, test_data):
    # your code here
    pass



