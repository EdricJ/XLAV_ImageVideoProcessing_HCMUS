import torch


class Loss:
    # Mean Square Error loss function
    @staticmethod
    def mse(y_true, y_pred):
        return torch.mean(torch.pow(y_true - y_pred, 2))
