class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    # computes the output Y of a lyaer for a given input X
    def forward_propagation(selfself, input):
        raise NotImplementedError

    # computes De/dX for a given De/dY (and update parametres if any)
    def backward_propagation(selfself, output_error, learning_rate):
        raise NotImplementedError

