#!/usr/bin/python

####################################################
# CS 5600/6600/7890: Assignment 1: Problems 1 & 2
# Erick Stone
# A02217762
#####################################################

import numpy as np


class and_perceptron:

    def __init__(self):
        # your code here
        self.bias = np.array([0])
        self.weights = np.array([0.26, 0.26])
        pass

    def output(self, x):
        # your code here
        # use rounding as an activation function
        return np.round(np.dot(x, self.weights) + self.bias)


class or_perceptron:
    def __init__(self):
        # your code
        self.bias = np.array([0.0])
        self.weights = np.array([0.51, 0.51])
        pass

    def output(self, x):
        # your code
        # use rounding as an activation function
        return np.round(np.dot(x, self.weights) + self.bias)


class not_perceptron:
    def __init__(self):
        # your code
        self.bias = np.array([1.0])
        self.weights = np.array([-1.0])
        pass

    def output(self, x):
        # your code
        # use rounding as an activation function
        return np.round(np.dot(x, self.weights) + self.bias)


class xor_perceptron:
    def __init__(self):
        # your code
        self.and_ = and_perceptron()
        self.or_ = or_perceptron()
        self.not_ = not_perceptron()
        pass

    def output(self, x):
        # your code
        # Not(A&B) & (A|B)
        left = self.not_.output(self.and_.output(x))
        right = self.or_.output(x)

        return self.and_.output([left[0], right[0]])


class xor_perceptron2:
    def __init__(self):
        # your code
        self.b1 = np.array([-1.0])
        self.b2 = np.array([-1.0])
        self.b3 = np.array([0])
        self.w1 = np.array([0.6, 0.51])
        self.w2 = np.array([1.1, 1.1])
        self.w3 = np.array([-1.5, 1.01])
        pass

    def output(self, x):
        # your code
        n1 = np.dot(x, self.w1) + self.b1
        n2 = np.dot(x, self.w2) + self.b2
        n3 = np.dot([n1[0], n2[0]], self.w3) + self.b3
        # activation function to reject numbers outside of 0 to 1
        return np.round(n3) if n3[0] < 1 else np.array([0])


### ================ Unit Tests ====================

# let's define a few binary input arrays.    
x00 = np.array([0, 0])
x01 = np.array([0, 1])
x10 = np.array([1, 0])
x11 = np.array([1, 1])


# let's test the and perceptron.
def unit_test_01():
    andp = and_perceptron()
    assert andp.output(x00) == 0
    assert andp.output(x01) == 0
    assert andp.output(x10) == 0
    assert andp.output(x11) == 1
    print('all andp assertions passed...')


# let's test the or perceptron.
def unit_test_02():
    orp = or_perceptron()
    assert orp.output(x00) == 0
    assert orp.output(x01) == 1
    assert orp.output(x10) == 1
    assert orp.output(x11) == 1
    print('all orp assertions passed...')


# let's test the not perceptron.
def unit_test_03():
    notp = not_perceptron()
    assert notp.output(np.array([0])) == 1
    assert notp.output(np.array([1])) == 0
    print('all notp assertions passed...')


# let's test the 1st xor perceptron.
def unit_test_04():
    xorp = xor_perceptron()
    assert xorp.output(x00) == 0
    assert xorp.output(x01) == 1
    assert xorp.output(x10) == 1
    assert xorp.output(x11) == 0
    print('all xorp assertions passed...')


# let's test the 2nd xor perceptron.
def unit_test_05():
    xorp2 = xor_perceptron2()
    assert xorp2.output(x00)[0] == 0
    assert xorp2.output(x01)[0] == 1
    assert xorp2.output(x10)[0] == 1
    assert xorp2.output(x11)[0] == 0
    print('all xorp2 assertions passed...')


unit_test_01()
unit_test_02()
unit_test_03()
unit_test_04()
unit_test_05()
