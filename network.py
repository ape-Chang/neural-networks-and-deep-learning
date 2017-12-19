#!/usr/bin/env python3

import numpy as np
import random

import loader


def sigmoid(z):
    def sigma0(x):
        return 1 / np.exp(x)

    def sigma1(x):
        # first attempt of optimization
        x = np.clip(x, -500, 500)
        return 1.0 / np.exp(-x)

    def sigma2(x):
        # second attempt of optimization
        # let's vectorize it
        # try to fix the overflow problem
        # and the divide by zero problem
        positive = np.copy(x)
        positive[positive < 0] = 0
        positive = np.exp(-positive)
        positive = 1 / (1 + positive)

        negative = np.copy(x)
        negative[negative > 0] = 0
        negative = np.exp(negative)
        negative = negative / (1 + negative)

        return positive * negative * 2

    return sigma2(z)


def sigmoid_prime(z):
    """ derivative of sigmoid """
    return sigmoid(z) * (1 - sigmoid(z))


class NeuralNetwork(object):
    def __init__(self, sizes, eta=0.01):
        # the construct of this network, sizes[0] means the number of input neurons,
        # sizes[-1] means the number of output neurons.
        self.sizes = sizes
        # the learning rate
        self.eta = eta
        self.biases = [np.random.randn(k, 1) for k in sizes[1:]]
        self.weights = [np.random.randn(k, n) for n, k in zip(sizes[:-1], sizes[1:])]

    def feed_forward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def back_propagate(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # activation of each layer
        activations = [x]
        z_values = []
        # forward pass
        a = x
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, a)
            a = sigmoid(z)

            z_values.append(z)
            activations.append(a)
        # backward pass
        delta = None
        for k in range(1, len(self.sizes)):
            delta = (activations[-k] - y if k == 1
            else np.dot(self.weights[-k + 1].transpose(), delta)) * sigmoid_prime(z_values[-k])
            nabla_b[-k] = delta
            nabla_w[-k] = np.dot(delta, activations[-k - 1].transpose())
        return nabla_b, nabla_w

    def update_mini_batch(self, batch):
        # nabla is name of the gradient operator
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in batch:
            delta_nabla_b, delta_nabla_w = self.back_propagate(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        m = len(batch)
        self.biases = [b - self.eta / m * nb for b, nb in zip(self.biases, nabla_b)]
        self.weights = [w - self.eta / m * nw for w, nw in zip(self.weights, nabla_w)]

    def sgd(self, training_data, epochs=5, batch_size=100):
        for epoch in range(epochs):
            self.sgd_one_epoch(training_data, batch_size)
            print("Epoch {} complete!".format(epoch))

    def sgd_one_epoch(self, training_data, batch_size):
        random.shuffle(training_data)
        for k in range(0, len(training_data), batch_size):
            self.update_mini_batch(training_data[k:k + batch_size])


if __name__ == '__main__':
    loader = loader.MNISTDataLoader()
    training_data, test_data = loader.load()
    network = NeuralNetwork([28 * 28, 56, 10])
    network.sgd(training_data)
