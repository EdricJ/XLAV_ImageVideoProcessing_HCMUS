# import required libraries
# import PyTorch
import torch
# import NumPy
import numpy as np
# numpy array
x_np = np.array([[1, 0, 2], [2, 0, 1]])
# PyTorch tensor from numpy array
x_torch = torch.from_numpy(x_np)
print('x_np', x_np)
print('x_torch', x_torch)
x_np += 1
print('x_np', x_np)
print('x_torch', x_torch)
x_torch += 1
print('x_np', x_np)
print('x_torch', x_torch)
# PyTorch tensor
y_torch = torch.tensor(([0, 8], [0, 4], [20, 20]),
dtype=torch.float)
# numpy array from PyTorch tensor
y_np = y_torch.numpy()
# or more explicit
y_np_cpu = y_torch.detach().cpu().numpy()
print('y_torch', y_torch)
print('y_np', y_np)
y_np += 1
print('y_torch', y_torch)
print('y_np', y_np)
y_torch += 1
print('y_torch', y_torch)
print('y_np', y_np)
