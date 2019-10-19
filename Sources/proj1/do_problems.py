import random

import numpy as np
import tflearn
from tflearn import regression, fully_connected, input_data

from proj1.get_data import get_data, combine_and_merge
from proj1.networks import build_1


def bee1():
    train = combine_and_merge('BEE1/bee_train/', 'BEE1/no_bee_train/')
    # test = combine_and_merge('BEE1/bee_test/', 'BEE1/no_bee_test/')
    # valid = combine_and_merge('BEE1/bee_valid/', 'BEE1/no_bee_valid/')
    model = build_1()
    model.fit(list(train[0]), list(train[1]), show_metric=True, n_epoch=40)
    print(model.evaluate(valid[0], valid[1]))


def bee2():
    train = get_data('BEE1/bee_train/', 1) + get_data('BEE1/no_bee_train/', 0)
    test = get_data('BEE1/bee_test/', 1) + get_data('BEE1/no_bee_test/', 0)
    valid = get_data('BEE1/bee_valid/', 1) + get_data('BEE1/no_bee_valid/', 0)


def buzz1():
    train = get_data('BEE1/bee_train/', 1) + get_data('BEE1/no_bee_train/', 0)
    test = get_data('BEE1/bee_test/', 1) + get_data('BEE1/no_bee_test/', 0)
    valid = get_data('BEE1/bee_valid/', 1) + get_data('BEE1/no_bee_valid/', 0)


def buzz2():
    train = get_data('BEE1/bee_train/', 1) + get_data('BEE1/no_bee_train/', 0)
    test = get_data('BEE1/bee_test/', 1) + get_data('BEE1/no_bee_test/', 0)
    valid = get_data('BEE1/bee_valid/', 1) + get_data('BEE1/no_bee_valid/', 0)

bee1()