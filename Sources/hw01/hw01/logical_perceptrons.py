#!/usr/bin/python

####################################################
# CS 5600/6600/7890: Assignment 1: Problems 1 & 2
# YOUR NAME
# YOUR A#
#####################################################

import numpy as np

class and_perceptron:

    def __init__(self):
        # your code here
        pass

    def output(self, x):
        # your code here
        pass

class or_perceptron:
    def __init__(self):
        # your code
        pass

    def output(self, x):
        # your code
        pass

class not_perceptron:
    def __init__(self):
        # your code
        pass

    def output(self, x):
        # your code
        pass

class xor_perceptron:
    def __init__(self):
        # your code
        pass

    def output(self, x):
        # your code
        pass


class xor_perceptron2:
    def __init__(self):
        # your code
        pass

    def output(self, x):
        # your code
        pass

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
    print 'all andp assertions passed...'

# let's test the or perceptron.
def unit_test_02():
    orp = or_perceptron()
    assert orp.output(x00) == 0
    assert orp.output(x01) == 1
    assert orp.output(x10) == 1
    assert orp.output(x11) == 1
    print 'all orp assertions passed...'

# let's test the not perceptron.
def unit_test_03():
    notp = not_perceptron()
    assert notp.output(np.array([0])) == 1
    assert notp.output(np.array([1])) == 0
    print 'all notp assertions passed...'

# let's test the 1st xor perceptron.
def unit_test_04():
    xorp = xor_perceptron()
    assert xorp.output(x00) == 0
    assert xorp.output(x01) == 1
    assert xorp.output(x10) == 1
    assert xorp.output(x11) == 0
    print 'all xorp assertions passed...'

# let's test the 2nd xor perceptron.
def unit_test_05():
    xorp2 = xor_perceptron2()
    assert xorp2.output(x00)[0] == 0
    assert xorp2.output(x01)[0] == 1
    assert xorp2.output(x10)[0] == 1
    assert xorp2.output(x11)[0] == 0
    print 'all xorp2 assertions passed...'

 
    
        
        

    
        





