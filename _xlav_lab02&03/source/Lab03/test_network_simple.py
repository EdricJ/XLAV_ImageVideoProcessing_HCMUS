import torch

from layer_simple import FCLayer, ActivationLayer
from function_simple import Activation, ActivationPrime
from function_simple import Loss, LossPrime
from network_simple import Network

# training data
x_train = torch.tensor([[[0, 0, 1, 0]], [[0, 1, 0, 1]], [[1, 0, 1, 1]], [[1, 1, 0, 0]], [[1, 1, 1, 1]]], dtype=torch.float)
y_train = torch.tensor([[[0]], [[1]], [[1]], [[0]], [[2]]], dtype=torch.float)

# network architecture
net = Network()
net.add(FCLayer(4, 5))
net.add(ActivationLayer(Activation.tanh, ActivationPrime.tanh_derivative))
net.add(FCLayer(5, 1))
net.add(ActivationLayer(Activation.tanh, ActivationPrime.tanh_derivative))

# train your network
net.use(Loss.mse, LossPrime.mse_prime)
net.fit(x_train, y_train, epochs=150, alpha=0.1)

# test
out = net.predicts(x_train)
print(out)
