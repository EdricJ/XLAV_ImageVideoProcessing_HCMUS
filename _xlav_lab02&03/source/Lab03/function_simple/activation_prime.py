import torch


class ActivationPrime:
    # derivative of sigmoid
    @staticmethod
    def sigmoid_derivative(s):
        return s * (1 - s)

    # derivative of tanh
    @staticmethod
    def tanh_derivative(s):
        return 1 - torch.tanh(s) ** 2
