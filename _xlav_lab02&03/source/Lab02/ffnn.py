# import PyTorch
import torch
# import PyTorch Neural Network module
import torch.nn as nn


# sigmoid activation
def sigmoid(s):
    return 1 / (1 + torch.exp(-s))


# derivative of sigmoid
def sigmoid_derivative(s):
    return s * (1 - s)


# Feed Forward Neural Network class
class FFNeuralNetwork(nn.Module):
    # initialization function
    def __init__(self, ):
        # init function of base class
        super(FFNeuralNetwork, self).__init__()

        # corresponding size of each layer
        self.inputSize = 4
        self.hiddenSize = 5
        self.outputSize = 1

        # random weights from a normal distribution
        self.W1 = torch.randn(self.inputSize, self.hiddenSize)  # 4 X 5 tensor
        self.W2 = torch.randn(self.hiddenSize, self.outputSize)  # 5 X 1 tensor

        self.z = None
        self.z_activation = None
        self.z_activation_derivative = None

        self.z2 = None
        self.z3 = None
        
        self.out_error = None
        self.out_delta = None

        self.z2_error = None
        self.z2_delta = None


    # activation function using sigmoid
    def activation(self, z):
        self.z_activation = sigmoid(z)
        return self.z_activation

    # derivative of activation function
    def activation_derivative(self, z):
        self.z_activation_derivative = sigmoid_derivative(z)
        return self.z_activation_derivative

    # forward propagation
    def forward(self, X):
        # multiply input X and weights W1 from input layer to hidden layer
        self.z = torch.matmul(X, self.W1)
        self.z2 = self.activation(self.z)  # activation function
        # multiply current tensor and weights W2 from hidden layer to output layer
        self.z3 = torch.matmul(self.z2, self.W2) 
        o = self.activation(self.z3)  # final activation function
        return o

    # backward propagation
    def backward(self, X, y, o, rate):
        self.out_error = y - o  # error in output
        self.out_delta = self.out_error * self.activation_derivative(o)  # derivative of activation to error

        # error and derivative of activation to error of next layer in backward propagation
        self.z2_error = torch.matmul(self.out_delta, torch.t(self.W2))
        self.z2_delta = self.z2_error * self.activation_derivative(self.z2)

        # update weights from delta of error and learning rate
        self.W1 += torch.matmul(torch.t(X), self.z2_delta) * rate 
        self.W2 += torch.matmul(torch.t(self.z2), self.out_delta) * rate

    # training function with learning rate parameter
    def train(self, X, y, rate):
        # forward + backward pass for training
        o = self.forward(X)
        
        self.backward(X, y, o, rate)

    # save weights of model
    @staticmethod
    def save_weights(model, path):
        # use the PyTorch internal storage functions
        torch.save(model, path)

    # load weights of model
    @staticmethod
    def load_weights(path):
        # reload model with all the weights
        torch.load(path)

    # predict function
    def predict(self, x_predict):
        print("Predict data based on trained weights: ")
        print("Input: \n" + str(x_predict))
        print("Output: \n" + str(self.forward(x_predict)))
