# /usr/bin/python

################################################################
# Erick Stone
# A02217762
# Write your code at the end of this file in the provided
# function stubs.
#
# Note: Put parens around print(statements if you're using Py3.
################################################################

#### Libraries
# Standard library
import json
import random
import sys
import pickle
from multiprocessing import Pool

sys.path.append('hw04')

# Third-party libraries
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import hw04.mnist_loader as ml


# Define the quadratic and cross-entropy cost functions

class QuadraticCost(object):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.

        """
        return 0.5 * np.linalg.norm(a - y) ** 2

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer."""
        return (a - y) * sigmoid_prime(z)


class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).

        """
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer.  Note that the
        parameter ``z`` is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta method for other cost classes.

        """
        return (a - y)


#### Main Network class
class Network(object):

    def __init__(self, sizes, cost=CrossEntropyCost):
        """The list ``sizes`` contains the number of neurons in the respective
        layers of the network.  For example, if the list was [2, 3, 1]
        then it would be a three-layer network, with the first layer
        containing 2 neurons, the second layer 3 neurons, and the
        third layer 1 neuron.  The biases and weights for the network
        are initialized randomly, using
        ``self.init_weights`` (see docstring for that
        method).

        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.init_weights()
        self.cost = cost

    ## normalized weight initializer
    def sqrt_norm_init_weights(self):
        """Initialize random weights with a standard deviation of 1/sqrt(x).
        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    ## large weight initializer
    def init_weights(self):
        """Initialize random weights.
        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda=0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False):
        """Train the neural network using mini-batch stochastic gradient
        descent.  The ``training_data`` is a list of tuples ``(x, y)``
        representing the training inputs and the desired outputs.  The
        other non-optional parameters are self-explanatory, as is the
        regularization parameter ``lmbda``.  The method also accepts
        ``evaluation_data``, usually either the validation or test
        data.  We can monitor the cost and accuracy on either the
        evaluation data or the training data, by setting the
        appropriate flags.  The method returns a tuple containing four
        lists: the (per-epoch) costs on the evaluation data, the
        accuracies on the evaluation data, the costs on the training
        data, and the accuracies on the training data.  All values are
        evaluated at the end of each training epoch.  So, for example,
        if we train for 30 epochs, then the first element of the tuple
        will be a 30-element list containing the cost on the
        evaluation data at the end of each epoch. Note that the lists
        are empty if the corresponding flag is not set.

        """
        if evaluation_data:
            n_eval_data = len(evaluation_data)
        n_train_data = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        prev_accuracy = 0.0
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n_train_data, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(
                    mini_batch, eta, lmbda, len(training_data))
            print("Epoch %s training complete" % j)
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print("Cost on training data: {}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy / float(n_train_data))
                print("Accuracy on training data: {} / {}".format(
                    accuracy, n_train_data))
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print("Cost on evaluation data: {}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy / float(n_eval_data))
                print("Accuracy on evaluation data: {} / {}".format(
                    self.accuracy(evaluation_data), n_eval_data))
        #         r = self.accuracy(evaluation_data) / n_eval_data
        #         # thresholded early stopping
        #         if r > prev_accuracy:
        #             if r - prev_accuracy < 0.01:
        #                 break
        #             prev_accuracy = self.accuracy(evaluation_data) / n_eval_data
        # while len(evaluation_accuracy) < epochs:
        #     evaluation_cost.append(0.0)
        #     evaluation_accuracy.append(0.0)
        #     training_cost.append(0.0)
        #     training_accuracy.append(0.0)
        return evaluation_cost, evaluation_accuracy, \
               training_cost, training_accuracy

    ## vladimir kulyukin 14may2018: same as above but
    ## the accuracy function is called with convert=True always
    ## to accomodate the bee data.
    def SGD2(self, training_data, epochs, mini_batch_size, eta,
             lmbda=0.0,
             evaluation_data=None,
             monitor_evaluation_cost=False,
             monitor_evaluation_accuracy=False,
             monitor_training_cost=False,
             monitor_training_accuracy=False):
        if evaluation_data:
            n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(
                    mini_batch, eta, lmbda, len(training_data))
            # print("Epoch %s training complete" % j
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print("Cost on training data: {}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print("Accuracy on training data: {} / {}".format(
                    accuracy, n))
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print("Cost on evaluation data: {}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data, convert=True)
                evaluation_accuracy.append(accuracy)
                # vladimir kulyukin: commented out
                # print("Accuracy on evaluation data: {} / {}".format(
                #    accuracy, n)
            # print
        return evaluation_cost, evaluation_accuracy, \
               training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        """Update the network's weights and biases by applying gradient
        descent using backpropagation to a single mini batch.  The
        ``mini_batch`` is a list of tuples ``(x, y)``, ``eta`` is the
        learning rate, ``lmbda`` is the regularization parameter, and
        ``n`` is the total size of the training data set.

        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [(1 - eta * (lmbda / n)) * w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass: zs[-1] is not used.
        # activations[-1] - y = (a - y).
        delta = (self.cost).delta(zs[-1], activations[-1], y)
        ## delta = (a^{L}_{j} - y_{j})
        nabla_b[-1] = delta
        ## nabla_w = a^{L-1}_{k}(a^{L}_{j} - y_{j}).
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def accuracy(self, data, convert=False):
        """Return the number of inputs in ``data`` for which the neural
        network outputs the correct result. The neural network's
        output is assumed to be the index of whichever neuron in the
        final layer has the highest activation.

        The flag ``convert`` should be set to False if the data set is
        validation or test data (the usual case), and to True if the
        data set is the training data. The need for this flag arises
        due to differences in the way the results ``y`` are
        represented in the different data sets.  In particular, it
        flags whether we need to convert between the different
        representations.  It may seem strange to use different
        representations for the different data sets.  Why not use the
        same representation for all three data sets?  It's done for
        efficiency reasons -- the program usually evaluates the cost
        on the training data and the accuracy on other data sets.
        These are different types of computations, and using different
        representations speeds things up.  More details on the
        representations can be found in
        mnist_loader.load_data_wrapper.

        """
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y)
                       for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)

    def total_cost(self, data, lmbda, convert=False):
        """Return the total cost for the data set ``data``.  The flag
        ``convert`` should be set to False if the data set is the
        training data (the usual case), and to True if the data set is
        the validation or test data.  See comments on the similar (but
        reversed) convention for the ``accuracy`` method, above.
        """
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert: y = vectorized_result(y)
            cost += self.cost.fn(a, y) / len(data)
        cost += 0.5 * (lmbda / len(data)) * sum(
            np.linalg.norm(w) ** 2 for w in self.weights)
        return cost

    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()


#### Loading a Network
def load(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network.

    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net


#### Miscellaneous functions
def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the j'th position
    and zeroes elsewhere.  This is used to convert a digit (0...9)
    into a corresponding desired output from the neural network.

    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))


def plot_costs(eval_costs, train_costs, num_epochs):
    # your code here
    y1 = eval_costs  # green
    y2 = train_costs  # blue
    fig, ax = plt.subplots()

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cost')
    ax.plot(list(range(0, num_epochs)), y1, 'green')
    ax.plot(list(range(0, num_epochs)), y2, 'blue')
    plt.show()
    pass


def plot_accuracies(eval_accs, train_accs, num_epochs):
    # your code here
    y1 = eval_accs  # green
    y2 = train_accs  # blue
    fig, ax = plt.subplots()

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cost')
    ax.plot(list(range(0, num_epochs)), y1, 'green')
    ax.plot(list(range(0, num_epochs)), y2, 'blue')
    plt.show()
    pass


## num_nodes -> (eval_cost, eval_acc, train_cost, train_acc)
## use this function to compute the eval_acc and min_cost.
def collect_1_hidden_layer_net_stats(lower_num_hidden_nodes,
                                     upper_num_hidden_nodes,
                                     cost_function,
                                     num_epochs,
                                     mbs,
                                     eta,
                                     lmbda,
                                     train_data,
                                     eval_data):
    dict = {}
    for i in range(lower_num_hidden_nodes, upper_num_hidden_nodes + 1):
        dict[i] = Network(
            [784, i, 10], cost=cost_function).SGD(
            train_data, num_epochs, mbs, eta, lmbda=lmbda,
            evaluation_data=eval_data,
            monitor_evaluation_cost=True,
            monitor_evaluation_accuracy=True,
            monitor_training_cost=True,
            monitor_training_accuracy=True)
    return dict


def collect_2_hidden_layer_net_stats(lower_num_hidden_nodes,
                                     upper_num_hidden_nodes,
                                     cost_function,
                                     num_epochs,
                                     mbs,
                                     eta,
                                     lmbda,
                                     train_data,
                                     eval_data):
    dict = {}
    for i in range(lower_num_hidden_nodes, upper_num_hidden_nodes + 1):
        for j in range(lower_num_hidden_nodes, upper_num_hidden_nodes + 1):
            dict[str(i) + '_' + str(j)] = Network(
                [784, i, j, 10], cost=cost_function).SGD(
                train_data, num_epochs, mbs, eta, lmbda=lmbda,
                evaluation_data=eval_data,
                monitor_evaluation_cost=True,
                monitor_evaluation_accuracy=True,
                monitor_training_cost=True,
                monitor_training_accuracy=True)
    return dict


def collect_3_hidden_layer_net_stats(lower_num_hidden_nodes,
                                     upper_num_hidden_nodes,
                                     cost_function,
                                     num_epochs,
                                     mbs,
                                     eta,
                                     lmbda,
                                     train_data,
                                     eval_data):
    dict = {}
    for i in range(lower_num_hidden_nodes, upper_num_hidden_nodes + 1):
        for j in range(lower_num_hidden_nodes, upper_num_hidden_nodes + 1):
            for k in range(lower_num_hidden_nodes, upper_num_hidden_nodes + 1):
                dict[str(i) + '_' + str(j) + '_' + str(k)] = Network(
                    [784, i, j, 10], cost=cost_function).SGD(
                    train_data, num_epochs, mbs, eta, lmbda=lmbda,
                    evaluation_data=eval_data,
                    monitor_evaluation_cost=True,
                    monitor_evaluation_accuracy=True,
                    monitor_training_cost=True,
                    monitor_training_accuracy=True)
    return dict


# train_d, valid_d, test_d = ml.load_data_wrapper()
# len(train_d)
# len(valid_d)
# len(test_d)
#
# if __name__ == '__main__':
#     net1 = Network([784, 71, 10], cost=CrossEntropyCost)
#     net2 = Network([784, 99, 99, 10], cost=CrossEntropyCost)
#     net3 = Network([784, 99, 99, 99, 10], cost=CrossEntropyCost)
#
#     net1.SGD(train_d, 30, 10, 0.25, lmbda=0.5,
#              evaluation_data=valid_d,
#              monitor_evaluation_cost=True,
#              monitor_evaluation_accuracy=True,
#              monitor_training_cost=True,
#              monitor_training_accuracy=True)
#
#     net2.SGD(train_d, 30, 10, 0.5, lmbda=0.5,
#              evaluation_data=valid_d,
#              monitor_evaluation_cost=True,
#              monitor_evaluation_accuracy=True,
#              monitor_training_cost=True,
#              monitor_training_accuracy=True)
#
#     net3.SGD(train_d, 30, 10, 0.4, lmbda=0.4,
#              evaluation_data=valid_d,
#              monitor_evaluation_cost=True,
#              monitor_evaluation_accuracy=True,
#              monitor_training_cost=True,
#              monitor_training_accuracy=True)
#
#     with open('net1.pck', 'wb') as f:
#         pickle.dump(net1, f)
#
#     with open('net2.pck', 'wb') as f:
#         pickle.dump(net2, f)
#
#     with open('net3.pck', 'wb') as f:
#         pickle.dump(net3, f)

#### COMMENTS:
# Network 1 performance:
# Cost on training data: 0.23366812190283776
# Accuracy on training data: 49695 / 50000
# Cost on evaluation data: 1.0814890753787079
# Accuracy on evaluation data: 9647 / 10000

# Network 2 performance:
# Cost on training data: 0.1928469441493975
# Accuracy on training data: 49725 / 50000
# Cost on evaluation data: 0.9902834236675866
# Accuracy on evaluation data: 9695 / 10000

# Network 3 performance:
# Cost on training data: 0.24608339408300342
# Accuracy on training data: 49681 / 50000
# Cost on evaluation data: 1.2828853347872249
# Accuracy on evaluation data: 9670 / 10000
