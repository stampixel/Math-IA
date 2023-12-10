import numpy as np


# loss function and its derivative
def mse(y_true, y_pred):
    """
    true value vs predicted value
    :param y_true:
    :param y_pred:
    :return:
    """
    return np.mean(np.power(y_true - y_pred, 2))


def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size
