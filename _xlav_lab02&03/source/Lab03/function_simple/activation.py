import torch


class Activation:
    # sigmoid activation
    @staticmethod
    def sigmoid(s):
        return 1 / (1 + torch.exp(-s))

    # tanh activation
    @staticmethod
    def tanh(s):
        return torch.tanh(s)
