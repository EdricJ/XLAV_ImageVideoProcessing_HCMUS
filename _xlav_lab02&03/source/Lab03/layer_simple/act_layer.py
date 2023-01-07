from layer_simple.base_layer import BaseLayer


class ActivationLayer(BaseLayer):
    def __init__(self, activation, activation_derivative):
        self.in_data = None
        self.out_data = None

        self.activation = activation
        self.activation_derivative = activation_derivative

    def forward(self, in_data):
        self.in_data = in_data
        self.out_data = self.activation(in_data)

        return self.out_data

    def backward(self, out_error, rate):
        return self.activation_derivative(self.in_data) * out_error
