from proj1.get_data import get_data
from proj1.networks import *


def bee1():
    x_train, y_train = get_data('BEE1/bee_train/', 'BEE1/no_bee_train/', 32 * 32)
    x_test, y_test = get_data('BEE1/bee_test/', 'BEE1/no_bee_test/', 32 * 32)
    x_valid, y_valid = get_data('BEE1/bee_valid/', 'BEE1/no_bee_valid/', 32 * 32)
    model = build_1()
    model.fit(x_train, y_train, validation_set=(x_test, y_test), show_metric=True, n_epoch=40)
    print(model.evaluate(x_valid, y_valid))
    model.save('bee1')


def bee2():
    x_train, y_train = get_data3('one_super/training/bee/', 'one_super/training/no_bee/', 150 * 150)
    x_test, y_test = get_data3('one_super/testing/bee/', 'one_super/testing/no_bee/', 150 * 150)
    x_valid, y_valid = get_data3('one_super/validation/bee/', 'one_super/validation/no_bee/', 150 * 150)
    model = build_2(150 * 150)
    model.fit(x_train, y_train, validation_set=(x_test, y_test), show_metric=True, n_epoch=130)
    print(model.evaluate(x_valid, y_valid))
    model.save('bee2')

    x_train, y_train = get_data3('two_super/training/bee/', 'one_super/training/no_bee/', 150 * 150)
    x_test, y_test = get_data3('two_super/testing/bee/', 'one_super/testing/no_bee/', 150 * 150)
    x_valid, y_valid = get_data3('two_super/validation/bee/', 'one_super/validation/no_bee/', 150 * 150)
    model = build_2(150 * 150)
    model.fit(x_train, y_train, validation_set=(x_test, y_test), show_metric=True, n_epoch=130)
    print(model.evaluate(x_valid, y_valid))
    model.save('bee2_2')


def buzz1():
    bee_x = get_wav('BEE1/bee_train/')
    c_x = get_wav('BEE1/bee_test/')
    n_x = get_wav('BEE1/bee_valid/')




def buzz2():
    train = get_data('BEE1/bee_train/', 1) + get_data('BEE1/no_bee_train/', 0)
    test = get_data('BEE1/bee_test/', 1) + get_data('BEE1/no_bee_test/', 0)
    valid = get_data('BEE1/bee_valid/', 1) + get_data('BEE1/no_bee_valid/', 0)


bee1()