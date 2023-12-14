import numpy as np

from network import Network
from main_layer import FCLayer
from activation import ActivationLayer
from activation import tanh, tanh_prime, relu, relu_prime
from mse_loss import mse, mse_prime

import pandas as pd
df = pd.read_csv('data.csv')

x_train = []
y_train = []


for i, row in df.iterrows():
    # x_train = np.append(x_train, np.array([[df.at[i, 'X-axis'], df.at[i, 'Y-axis']]]))
    x_train.append([[df.at[i, 'X-axis'], df.at[i, 'Y-axis']]])
    if df.at[i, 'Cluster'] == 0:
        y_train.append([[1, 0, 0]])

    elif df.at[i, 'Cluster'] == 1:
        y_train.append([[0, 1, 0]])

    elif df.at[i, 'Cluster'] == 2:
        y_train.append([[0, 0, 1]])

    # y_train.append([[df.at[i, 'Cluster']]])

x_train = np.array(x_train)
y_train = np.array(y_train)
print(x_train)
print(y_train)


# training data


# network
net = Network()
net.add(FCLayer(2, 4))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(4, 4))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(4, 3))
net.add(ActivationLayer(tanh, tanh_prime))

# train
net.use(mse, mse_prime)
net.fit(x_train, y_train, epochs=1000, learning_rate=0.1)

# test
out = net.predict(x_train)
print(out)
