import torch


class LossPrime:
    # derivative of Mean Square Error loss function
    @staticmethod
    def mse_prime(y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.numel()
