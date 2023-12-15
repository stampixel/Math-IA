from layer import Layer
import numpy as np


# inherit from base class Layer
class FCLayer(Layer):
    # input_size = number of input neurons
    # ouput_size = number of output neurons
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weights = np.random.rand(input_size, output_size) - 0.5
        print(f"weight: {self.weights}")
        self.bias = np.random.rand(1, output_size) - 0.5
        print(f"bias: {self.bias}")


    # return output for a given input
    def forward_propagation(self, input_data):
        # print("weights:", self.weights)
        # print("bias:", self.bias)

        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        # update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error

        return input_error
