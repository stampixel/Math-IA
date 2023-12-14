import numpy as np

from network import Network
from main_layer import FCLayer
from activation import ActivationLayer
from activation import tanh, tanh_prime, relu, relu_prime
from mse_loss import mse, mse_prime

import pandas as pd

df = pd.read_csv('datav2.csv')

df = df.sample(frac=1)

x_train = []
y_train = []

print(df)
for i, row in df.iterrows():
    # x_train = np.append(x_train, np.array([[df.at[i, 'X-axis'], df.at[i, 'Y-axis']]]))
    x_train.append([[round(df.at[i, 'X-axis']), round(df.at[i, 'Y-axis'])]])
    if df.at[i, 'Value'] == 1.0000:
        y_train.append([[1]])

    elif df.at[i, 'Value'] == -1.0000:
        y_train.append([[-1]])

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
net.add(FCLayer(4, 1))
net.add(ActivationLayer(tanh, tanh_prime))

# train
net.use(mse, mse_prime)
net.fit(x_train, y_train, epochs=1000, learning_rate=0.1)

# test
out = net.predict(x_train)
print(out)
print(y_train)
