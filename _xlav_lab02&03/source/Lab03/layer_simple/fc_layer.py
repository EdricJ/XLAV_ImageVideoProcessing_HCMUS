from layer_simple.base_layer import BaseLayer

import torch


class FCLayer(BaseLayer):
    def __init__(self, in_size, out_size):
        self.in_data = None
        self.out_data = None

        self.weights = torch.randn(in_size, out_size)
        self.bias = torch.randn(1, out_size)

    def forward(self, in_data):
        self.in_data = in_data
        self.out_data = torch.matmul(self.in_data, self.weights) + self.bias

        return self.out_data

    def backward(self, out_error, rate):
        in_error = torch.matmul(out_error, self.weights.T)
        weights_error = torch.matmul(self.in_data.T, out_error)

        self.weights -= rate * weights_error
        self.bias -= rate * out_error

        return in_error
