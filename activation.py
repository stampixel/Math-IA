from layer import Layer
import numpy as np


class ActivationLayer(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward_propagation(self, input_data):
        """
        returns the activated input
        :param input_data:
        :return:
        """
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        """
        returns input_error=De/dX for a given output_error=dE/dY
        :param output_error:
        :param learning_rate:
        :return:
        """
        return self.activation_prime(self.input) * output_error


def tanh(x):
    return np.tanh(x)


def relu(x):
    return np.maximum(x, 0)


def relu_prime(x):
    # x[x > 0] = 1
    # # return 0 if np.maximum(x, 0) == 0 else 1
    # return np.maximum(0, x)
    return np.where(x > 0, 1.0, 0.0)


def tanh_prime(x):
    return 1 - np.tanh(x) ** 2
