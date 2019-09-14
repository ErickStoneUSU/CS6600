#####################
# module: hw02_f19.py
# author: Erick Stone A02217762
#####################

import numpy as np
import pickle
from hw02.hw02_f19_data import *


# save() function to save the trained network to a file
def save(ann, file_name):
    with open(file_name, 'wb') as fp:
        pickle.dump(ann, fp)


# restore() function to restore the file
def load(file_name):
    with open(file_name, 'rb') as fp:
        nn = pickle.load(fp)
    return nn


def build_nn_wmats(mat_dims):
    # your code here
    lis = []

    for i in range(len(mat_dims) - 1):
        n1 = np.random.uniform(0, 1, [mat_dims[i], mat_dims[i + 1]])
        lis.append(n1)

    return tuple(lis)


def build_231_nn():
    return build_nn_wmats((2, 3, 1))


def build_2441_nn():
    return build_nn_wmats((2, 4, 4, 1))


def build_484_nn():
    return build_nn_wmats((4, 8, 4))


def build_4444_nn():
    return build_nn_wmats((4, 4, 4, 4))


def sig(x):
    return 1 / (1+(np.e**-x))


def d_sig(x):
    return x * (1-x)


def train_3_layer_nn(numIters, X, y, build):
    W1, W2 = build()
    for j in range(numIters):
        # Feedforward
        Z2 = np.dot(X, W1)
        a2 = sig(Z2)
        Z3 = np.dot(a2, W2)
        a3 = sig(Z3)

        # Backprop
        a3_error = y - a3
        a3_delta = a3_error * d_sig(a3)
        W2 += a2.T.dot(a3_delta)

        a2_error = a3_delta.dot(W2.T)
        a2_delta = a2_error * d_sig(a2)
        W1 += X.T.dot(a2_delta)
    return W1, W2


def train_4_layer_nn(numIters, X, y, build):
    W1, W2, W3 = build()
    for j in range(numIters):
        # Feedforward
        Z2 = np.dot(X, W1)
        a2 = sig(Z2)
        Z3 = np.dot(a2, W2)
        a3 = sig(Z3)
        Z4 = np.dot(a3, W3)
        a4 = sig(Z4)

        # Backprop
        a4_error = y - a4
        a4_delta = a4_error * d_sig(a4)
        W3 += a3.T.dot(a4_delta)

        a3_error = a4_delta.dot(W3.T)
        a3_delta = a3_error * d_sig(a3)
        W2 += a2.T.dot(a3_delta)

        a2_error = a3_delta.dot(W2.T)
        a2_delta = a2_error * d_sig(a2)
        W1 += X.T.dot(a2_delta)
    return W1, W2, W3


def fit_3_layer_nn(x, wmats, thresh=0.4, thresh_flag=False):
    # feed forward
    a = [x]
    for w in wmats:
        a.append(sig(np.dot(a[-1], w)))

    if thresh_flag:
        return sum(a[-1]) / len(a[-1]) > thresh

    return sum(a[-1]) / len(a[-1])


def fit_4_layer_nn(x, wmats, thresh=0.4, thresh_flag=False):
    return fit_3_layer_nn(x, wmats, thresh=0.4, thresh_flag=False)


# Remember to state in your comments the structure of each of your
# ANNs (e.g., 2 x 3 x 1 or 2 x 4 x 4 x 1) and how many iterations
# it took you to train it.

save(train_3_layer_nn(5000, X1, y_and, build_231_nn), 'and_3_layer_ann.pck')
save(train_3_layer_nn(5000, X1, y_or, build_231_nn), 'or_3_layer_ann.pck')
save(train_3_layer_nn(5000, X1, y_xor, build_231_nn), 'xor_3_layer_ann.pck')
save(train_3_layer_nn(5000, X2, y_not, build_231_nn), 'not_3_layer_ann.pck')
save(train_3_layer_nn(5000, X4, bool_exp, build_484_nn), 'bool_3_layer_ann.pck')

save(train_4_layer_nn(5000, X1, y_and, build_2441_nn), 'and_4_layer_ann.pck')
save(train_4_layer_nn(5000, X1, y_or, build_2441_nn), 'or_4_layer_ann.pck')
save(train_4_layer_nn(5000, X1, y_xor, build_2441_nn), 'xor_4_layer_ann.pck')
save(train_4_layer_nn(5000, X2, y_not, build_2441_nn), 'not_4_layer_ann.pck')
save(train_4_layer_nn(5000, X4, bool_exp, build_4444_nn), 'bool_4_layer_ann.pck')

# Assert to verify the contents of each of the pickled files
assert not fit_3_layer_nn(X1[0], load('and_3_layer_ann.pck'), thresh=0.59, thresh_flag=True)
assert not fit_3_layer_nn(X1[1], load('and_3_layer_ann.pck'), thresh=0.59, thresh_flag=True)
assert not fit_3_layer_nn(X1[2], load('and_3_layer_ann.pck'), thresh=0.59, thresh_flag=True)
assert fit_3_layer_nn(X1[3], load('and_3_layer_ann.pck'), thresh=0.59, thresh_flag=True)

assert not fit_3_layer_nn(X1[0], load('or_3_layer_ann.pck'), thresh=0.59, thresh_flag=True)
assert fit_3_layer_nn(X1[1], load('or_3_layer_ann.pck'), thresh=0.59, thresh_flag=True)
assert fit_3_layer_nn(X1[2], load('or_3_layer_ann.pck'), thresh=0.59, thresh_flag=True)
assert fit_3_layer_nn(X1[3], load('or_3_layer_ann.pck'), thresh=0.59, thresh_flag=True)

assert not fit_3_layer_nn(X1[0], load('xor_3_layer_ann.pck'), thresh=0.59, thresh_flag=True)
assert fit_3_layer_nn(X1[1], load('xor_3_layer_ann.pck'), thresh=0.59, thresh_flag=True)
assert fit_3_layer_nn(X1[2], load('xor_3_layer_ann.pck'), thresh=0.59, thresh_flag=True)
assert not fit_3_layer_nn(X1[3], load('xor_3_layer_ann.pck'), thresh=0.59, thresh_flag=True)

assert fit_3_layer_nn(X2[0], load('not_3_layer_ann.pck'), thresh=0.59, thresh_flag=True)
assert not fit_3_layer_nn(X2[1], load('not_3_layer_ann.pck'), thresh=0.59, thresh_flag=True)

assert fit_3_layer_nn(X4[0], load('bool_3_layer_ann.pck'), thresh=0.59, thresh_flag=True)
assert not fit_3_layer_nn(X4[1], load('bool_3_layer_ann.pck'), thresh=0.59, thresh_flag=True)
assert not fit_3_layer_nn(X4[2], load('bool_3_layer_ann.pck'), thresh=0.59, thresh_flag=True)
assert not fit_3_layer_nn(X4[3], load('bool_3_layer_ann.pck'), thresh=0.59, thresh_flag=True)
assert fit_3_layer_nn(X4[4], load('bool_3_layer_ann.pck'), thresh=0.59, thresh_flag=True)
assert not fit_3_layer_nn(X4[5], load('bool_3_layer_ann.pck'), thresh=0.59, thresh_flag=True)
assert not fit_3_layer_nn(X4[6], load('bool_3_layer_ann.pck'), thresh=0.59, thresh_flag=True)
assert not fit_3_layer_nn(X4[7], load('bool_3_layer_ann.pck'), thresh=0.59, thresh_flag=True)
assert fit_3_layer_nn(X4[8], load('bool_3_layer_ann.pck'), thresh=0.59, thresh_flag=True)
assert not fit_3_layer_nn(X4[9], load('bool_3_layer_ann.pck'), thresh=0.59, thresh_flag=True)
assert not fit_3_layer_nn(X4[10], load('bool_3_layer_ann.pck'), thresh=0.59, thresh_flag=True)
assert not fit_3_layer_nn(X4[11], load('bool_3_layer_ann.pck'), thresh=0.59, thresh_flag=True)
assert fit_3_layer_nn(X4[12], load('bool_3_layer_ann.pck'), thresh=0.59, thresh_flag=True)
assert fit_3_layer_nn(X4[13], load('bool_3_layer_ann.pck'), thresh=0.59, thresh_flag=True)
assert fit_3_layer_nn(X4[14], load('bool_3_layer_ann.pck'), thresh=0.59, thresh_flag=True)
assert fit_3_layer_nn(X4[15], load('bool_3_layer_ann.pck'), thresh=0.59, thresh_flag=True)

