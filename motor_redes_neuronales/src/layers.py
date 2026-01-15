"""
layers.py - Capas para Redes Neuronales
=======================================

Este módulo implementa las capas que componen una red neuronal.
Cada capa encapsula:
    - Parámetros entrenables (pesos W y sesgos b)
    - Lógica del forward pass (transformación lineal + activación)
    - Lógica del backward pass (cálculo de gradientes)
    - Inicialización de pesos (He, Xavier o aleatoria)

Capas implementadas:
    - Layer: Clase base abstracta
    - Dense: Capa completamente conectada (fully connected)

La arquitectura permite añadir fácilmente nuevos tipos de capas
heredando de la clase Layer.

Autores: Raúl Mendoza, Adrián Ojeda, Varela
Asignatura: Optimización y Heurística - ULPGC
"""

import numpy as np
from . import activations as act


class Layer:
    """
    Clase base abstracta para todas las capas de la red.
    
    Define la interfaz que deben implementar todas las capas:
        - forward(): Propagación hacia adelante
        - backward(): Retropropagación del gradiente
        - params(): Acceso a parámetros entrenables
        - grads(): Acceso a gradientes calculados
    
    Las subclases deben sobrescribir forward() y backward().
    """
    
    def forward(self, x):
        """
        Propagación hacia adelante.
        
        Parameters
        ----------
        x : numpy.ndarray
            Entrada a la capa.
            
        Returns
        -------
        numpy.ndarray
            Salida de la capa.
            
        Raises
        ------
        NotImplementedError
            Si la subclase no implementa este método.
        """
        raise NotImplementedError

    def backward(self, grad_output):
        """
        Retropropagación del gradiente.
        
        Parameters
        ----------
        grad_output : numpy.ndarray
            Gradiente de la pérdida respecto a la salida de esta capa.
            
        Returns
        -------
        numpy.ndarray
            Gradiente respecto a la entrada (para propagar a la capa anterior).
            
        Raises
        ------
        NotImplementedError
            Si la subclase no implementa este método.
        """
        raise NotImplementedError

    def params(self):
        """
        Devuelve los parámetros entrenables de la capa.
        
        Returns
        -------
        list
            Lista de arrays numpy con los parámetros (vacía por defecto).
        """
        return []

    def grads(self):
        """
        Devuelve los gradientes calculados de la capa.
        
        Returns
        -------
        list
            Lista de arrays numpy con los gradientes (vacía por defecto).
        """
        return []


class Dense(Layer):
    """
    Capa Densa (Fully Connected / Linear Layer).
    
    Implementa una capa completamente conectada donde cada neurona
    de entrada está conectada con cada neurona de salida mediante
    pesos entrenables.
    
    Operación matemática:
        z = x @ W + b           (transformación lineal)
        a = activation(z)       (función de activación)
    
    Durante el backward pass, calcula:
        - dL/dW: Gradiente respecto a los pesos
        - dL/db: Gradiente respecto a los sesgos
        - dL/dx: Gradiente respecto a la entrada (para propagar)
    
    Attributes
    ----------
    n_in : int
        Número de neuronas de entrada (dimensión del vector de entrada).
    n_out : int
        Número de neuronas de salida (dimensión del vector de salida).
    activation_name : str or None
        Nombre de la función de activación ('sigmoid', 'relu', 'tanh', 'softmax').
    W : numpy.ndarray
        Matriz de pesos de forma (n_in, n_out).
    b : numpy.ndarray
        Vector de sesgos de forma (1, n_out).
    dW : numpy.ndarray
        Gradiente de la pérdida respecto a W.
    db : numpy.ndarray
        Gradiente de la pérdida respecto a b.
    
    Examples
    --------
    >>> layer = Dense(784, 128, activation='relu', weight_init='he')
    >>> output = layer.forward(X)  # X de forma (batch, 784)
    >>> # output tiene forma (batch, 128)
    """
    
    def __init__(self, n_in, n_out, activation=None, weight_init="he"):
        """
        Inicializa la capa densa con pesos y sesgos.
        
        Parameters
        ----------
        n_in : int
            Dimensión de entrada (número de features/neuronas de entrada).
        n_out : int
            Dimensión de salida (número de neuronas en esta capa).
        activation : str, optional
            Función de activación: 'sigmoid', 'relu', 'tanh', 'softmax' o None.
            Si es None, no se aplica activación (capa lineal pura).
        weight_init : str, optional
            Método de inicialización de pesos:
            - 'he': Inicialización He (recomendado para ReLU)
            - 'xavier': Inicialización Xavier/Glorot (recomendado para sigmoid/tanh)
            - otro: Inicialización aleatoria pequeña (*0.01)
            
        Notes
        -----
        La inicialización de pesos es crítica para la convergencia:
        - He: W ~ N(0, sqrt(2/n_in)) - óptimo para ReLU
        - Xavier: W ~ N(0, sqrt(1/n_in)) - óptimo para sigmoid/tanh
        """
        self.n_in = n_in
        self.n_out = n_out
        self.activation_name = activation

        if weight_init == "he":
            self.W = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
        elif weight_init == "xavier":
            self.W = np.random.randn(n_in, n_out) * np.sqrt(1.0 / n_in)
        else:
            self.W = np.random.randn(n_in, n_out) * 0.01

        self.b = np.zeros((1, n_out))

        self._x = None
        self._z = None

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def _activation(self, z):
        """
        Aplica la función de activación.
        
        Parameters
        ----------
        z : numpy.ndarray
            Valores pre-activación (salida de la transformación lineal).
            
        Returns
        -------
        numpy.ndarray
            Valores post-activación.
        """
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
        """
        Calcula la derivada de la función de activación.
        
        Parameters
        ----------
        z : numpy.ndarray
            Valores pre-activación almacenados durante el forward pass.
            
        Returns
        -------
        numpy.ndarray
            Derivada de la activación evaluada en z.
            
        Notes
        -----
        Para softmax + cross-entropy, la derivada se maneja de forma
        combinada en la función de pérdida, por lo que aquí devolvemos 1.
        """
        if self.activation_name is None:
            return 1.0
        if self.activation_name == "sigmoid":
            return act.sigmoid_derivative(z)
        if self.activation_name == "relu":
            return act.relu_derivative(z)
        if self.activation_name == "tanh":
            return act.tanh_derivative(z)
        if self.activation_name == "softmax":
            return np.ones_like(z)
        raise ValueError(f"Derivada de activación {self.activation_name} no soportada")

    def forward(self, x):
        """
        Propagación hacia adelante de la capa.
        
        Realiza la transformación lineal seguida de la función de activación:
            z = x @ W + b
            a = activation(z)
        
        Parameters
        ----------
        x : numpy.ndarray
            Entrada de forma (batch_size, n_in).
            
        Returns
        -------
        numpy.ndarray
            Salida activada de forma (batch_size, n_out).
            
        Notes
        -----
        Los valores de entrada (x) y pre-activación (z) se almacenan
        en caché para usarse durante el backward pass.
        """
        self._x = x
        z = x @ self.W + self.b
        self._z = z
        return self._activation(z)

    def backward(self, grad_output):
        """
        Retropropagación del gradiente a través de la capa.
        
        Calcula los gradientes de la pérdida respecto a:
            - W (pesos): dL/dW = x^T @ grad_act
            - b (sesgos): dL/db = sum(grad_act, axis=0)
            - x (entrada): dL/dx = grad_act @ W^T (para propagar hacia atrás)
        
        Parameters
        ----------
        grad_output : numpy.ndarray
            Gradiente de la pérdida respecto a la salida de esta capa,
            de forma (batch_size, n_out). Es dL/da donde a es la activación.
            
        Returns
        -------
        numpy.ndarray
            Gradiente respecto a la entrada, de forma (batch_size, n_in).
            Este gradiente se pasa a la capa anterior.
            
        Notes
        -----
        Para softmax + cross-entropy, grad_output ya incluye la derivada
        combinada (y_pred - y_true)/batch, por lo que no multiplicamos
        por la derivada de softmax.
        """
        if self.activation_name in (None, "softmax"):
            grad_act = grad_output
        else:
            grad_act = grad_output * self._activation_derivative(self._z)

        self.dW = self._x.T @ grad_act
        
        self.db = np.sum(grad_act, axis=0, keepdims=True)

        grad_input = grad_act @ self.W.T
        return grad_input

    def params(self):
        """
        Devuelve los parámetros entrenables de la capa.
        
        Returns
        -------
        list
            [W, b] - Lista con la matriz de pesos y el vector de sesgos.
        """
        return [self.W, self.b]

    def grads(self):
        """
        Devuelve los gradientes calculados de los parámetros.
        
        Returns
        -------
        list
            [dW, db] - Lista con los gradientes de pesos y sesgos.
        """
        return [self.dW, self.db]
