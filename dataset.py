import numpy as np

from network import Network
from main_layer import FCLayer
from activation import ActivationLayer
from activation import tanh, tanh_prime, relu, relu_prime
from mse_loss import mse, mse_prime

import pandas as pd

df = pd.read_csv('data/data.csv')

x_train = []
y_train = []

for i, row in df.iterrows():
    # x_train = np.append(x_train, np.array([[df.at[i, 'X-axis'], df.at[i, 'Y-axis']]]))
    x_train.append([[df.at[i, 'X-axis'], df.at[i, 'Y-axis']]])
    if df.at[i, 'Cluster'] == 0:  # r
        y_train.append([[1, 0, 0]])

    elif df.at[i, 'Cluster'] == 1:  # g
        y_train.append([[0, 1, 0]])

    elif df.at[i, 'Cluster'] == 2:  # b
        y_train.append([[0, 0, 1]])

    # y_train.append([[df.at[i, 'Cluster']]])

x_train = np.array(x_train)
y_train = np.array(y_train)
# print(x_train)
# print(y_train)

# training data


# network
net = Network()
net.add(FCLayer(2, 4, np.array([[0.338, -0.418, -0.038, -0.263],
                                [0.152, 0.074, -0.381, -0.015]]), np.array([[-0.272, -0.273, 0.411, -0.113]])))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(4, 4, np.array([[-0.246, -0.323, -0.327, 0.418],
                                [0.422, 0.405, 0.191, 0.448],
                                [-0.445, -0.23, -0.44, -0.408],
                                [-0.312, 0.318, -0.21, -0.178]]), np.array([[0.216, -0.075, 0.158, -0.19]])))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(4, 3, np.array([[-0.107, -0.093, 0.327],
                                [-0.479, 0.404, -0.384],
                                [-0.22, -0.333, 0.336],
                                [-0.378, 0.366, 0.079]]), np.array([[-0.078, 0.195, 0.408]])))
net.add(ActivationLayer(tanh, tanh_prime))

# train
net.use(mse, mse_prime)
net.fit(x_train, y_train, epochs=1, learning_rate=0.1)

# test
out = net.predict(x_train)
print(out)

# print(net.predict(np.array([[[6.698483991405154, 6.4584774905415765],
#                              [6.725410456311336, 6.498399029655705],
#                              [9.849761950374077, 11.128270344485388]]])))
