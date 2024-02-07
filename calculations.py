import numpy as np
from mse_loss import mse, mse_prime

# Storing all randomly generated weights and biases into lists

l_1 = [[7.0, 6.0]] # This is the input
w_1 = [[0.338, -0.418, -0.038, -0.263],
       [0.152, 0.074, -0.381, -0.015]]
b_1 = [[-0.272, -0.273, 0.411, -0.113]]
w_2 = [[-0.246, -0.323, -0.327, 0.418],
       [0.422, 0.405, 0.191, 0.448],
       [-0.445, -0.23, -0.44, -0.408],
       [-0.312, 0.318, -0.21, -0.178]]
b_2 = [[0.216, -0.075, 0.158, -0.19]]
w_3 = [[-0.107, -0.093, 0.327],
       [-0.479, 0.404, -0.384],
       [-0.22, -0.333, 0.336],
       [-0.378, 0.366, 0.079]]
b_3 = [[-0.078, 0.195, 0.408]]

# print(np.tanh(-0.208 * 7 + (-0.008)))

output = np.tanh(np.dot(l_1, w_1) + b_1)
output = np.array([[round(num, 3) for num in output[i]] for i in range(len(output))])

print(output)

output = np.tanh(np.dot(output, w_2) + b_2)
output = np.array([[round(num, 3) for num in output[i]] for i in range(len(output))])

print(output)

output = np.tanh(np.dot(output, w_3) + b_3)
output = np.array([[round(num, 3) for num in output[i]] for i in range(len(output))])

print(output)


print(mse(y_true=[0, 1, 0], y_pred=output)),


print(mse_prime(np.array([[0, 1, 0]]), np.array([[ 0.044, -0.082,  0.71 ]])))

