import numpy as np

def mse(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def mse_grad(y_pred, y_true):
    return 2 * (y_pred - y_true) / y_true.size

def cross_entropy(y_pred, y_true, eps=1e-12):
    """
    y_true: one-hot (batch, n_classes)
    y_pred: probabilidades (batch, n_classes)
    """
    y_pred = np.clip(y_pred, eps, 1. - eps)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def cross_entropy_grad(y_pred, y_true):
    # derivada de softmax+crossentropy cuando el forward usó softmax
    # esto se simplifica a (y_pred - y_true) / batch
    batch_size = y_true.shape[0]
    return (y_pred - y_true) / batch_size
