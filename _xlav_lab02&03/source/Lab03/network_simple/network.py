from function_simple import Loss, LossPrime
import torch


class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    def add(self, layer):
        self.layers.append(layer)

    def use(self, loss: Loss, loss_prime: LossPrime) -> None:
        self.loss = loss
        self.loss_prime = loss_prime

    # forward propagation
    def predict(self, data):
        output = data
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def predicts(self, data):
        samples = len(data)
        result = []

        # for every input vector data x_i do:
        for i in range(samples):
            # forward propagation
            output = data[i]
            for layer in self.layers:
                output = layer.forward(output)
            result.append(output)
        return result

    def fit(self, x_train, y_train, epochs, alpha):
        samples = len(x_train)

        # for every training cycle -> forward() -> forward() -> Loss() <- backward() <- backward() ...
        for i in range(epochs):
            error = 0
            for k in range(samples):
                # forward propagation
                output = x_train[k]
                for layer in self.layers:
                    # output of the previous layer -> input to the current layer @l
                    output = layer.forward(output)
                # total error after current x_k has passed training cycle.
                error += self.loss(y_train[k], output)

                # Backward propagation
                gradient = self.loss_prime(y_train[k], output)
                for layer in reversed(self.layers):
                    gradient = layer.backward(gradient, alpha)

            # the total average error after one epoch?
            error /= samples
            print('On epoch ' + str(i + 1) + ' an average error = ' + str(error))
