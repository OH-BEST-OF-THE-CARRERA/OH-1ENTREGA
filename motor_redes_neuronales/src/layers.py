import numpy as np
from . import activations as act


class Layer:
    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError

    def params(self):
        return []

    def grads(self):
        return []


class Dense(Layer):
    def __init__(self, n_in, n_out, activation=None, weight_init="he"):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_name = activation

        # inicialización de pesos
        if weight_init == "he":
            self.W = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
        elif weight_init == "xavier":
            self.W = np.random.randn(n_in, n_out) * np.sqrt(1.0 / n_in)
        else:
            self.W = np.random.randn(n_in, n_out) * 0.01

        self.b = np.zeros((1, n_out))

        # caches para backward
        self._x = None
        self._z = None

        # gradientes
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def _activation(self, z):
        if self.activation_name is None:
            return z
        if self.activation_name == "sigmoid":
            return act.sigmoid(z)
        if self.activation_name == "relu":
            return act.relu(z)
        if self.activation_name == "tanh":
            return act.tanh(z)
        if self.activation_name == "softmax":
            return act.softmax(z)
        raise ValueError(f"Activación {self.activation_name} no soportada")

    def _activation_derivative(self, z):
        if self.activation_name is None:
            return 1.0
        if self.activation_name == "sigmoid":
            return act.sigmoid_derivative(z)
        if self.activation_name == "relu":
            return act.relu_derivative(z)
        if self.activation_name == "tanh":
            return act.tanh_derivative(z)
        # softmax normalmente se trata junto con la pérdida
        if self.activation_name == "softmax":
            # este caso suele manejarse fuera; devolvemos 1 para no romper
            return np.ones_like(z)
        raise ValueError(f"Derivada de activación {self.activation_name} no soportada")

    def forward(self, x):
        """
        x: (batch, n_in)
        """
        self._x = x
        z = x @ self.W + self.b  # (batch, n_out)
        self._z = z
        return self._activation(z)

    def backward(self, grad_output):
        """
        grad_output: dL/dA (batch, n_out)
        devuelve: dL/dX
        """
        # derivada de la activación
        if self.activation_name in (None, "softmax"):
            # en softmax + cross entropy el grad ya viene "hecho"
            grad_act = grad_output
        else:
            grad_act = grad_output * self._activation_derivative(self._z)

        # gradientes de pesos y bias
        # dL/dW = X^T * grad_act
        self.dW = self._x.T @ grad_act  # (n_in, batch) @ (batch, n_out) = (n_in, n_out)
        self.db = np.sum(grad_act, axis=0, keepdims=True)

        # dL/dX = grad_act * W^T
        grad_input = grad_act @ self.W.T
        return grad_input

    def params(self):
        return [self.W, self.b]

    def grads(self):
        return [self.dW, self.db]
