# import PyTorch
import torch
# import Feed Forward Neural Network class from nn_simple module
from ffnn import *

# sample input and output value for training
X = torch.tensor(([2, 9, 0, 7], [1, 5, 1, 8], [3, 6, 2, 1], [3, 1, 2, 9]), dtype=torch.float)  # 4 X 4 tensor    
y = torch.tensor(([90], [100], [88], [120]), dtype=torch.float)  # 4 X 1 tensor 

# scale units by max value
X_max, _ = torch.max(X, 0)
X = torch.div(X, X_max)
y = y / 120  # for max test score is 120

# sample input x for predicting
x_predict = torch.tensor(([3, 8, 4, 5]), dtype=torch.float)  # 1 X 4 tensor 

# scale input x by max value
x_predict_max, _ = torch.max(x_predict, 0)
x_predict = torch.div(x_predict, x_predict_max)

# create new object of implemented class
NN = FFNeuralNetwork()

# trains the NN 1,000 times
for i in range(1000):
    # print mean sum squared loss
    print("#" + str(i) + " Loss: " + str(torch.mean((y - NN(X)) ** 2).detach().item()))
    # training with learning rate = 0.1
    NN.train(X, y, 5)
# save weights
NN.save_weights(NN, "NN")

# load saved weights
NN.load_weights("NN")
# predict x input
NN.predict(x_predict)
