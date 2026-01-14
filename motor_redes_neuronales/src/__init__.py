"""
Motor de Redes Neuronales - Paquete Principal
==============================================

Este paquete implementa un motor de redes neuronales densas (Fully Connected
Neural Networks) desde cero, utilizando únicamente NumPy para las operaciones
numéricas.

Módulos incluidos:
    - network: Clase NeuralNetwork para construir y gestionar la red
    - layers: Capas de la red (Dense/Fully Connected)
    - activations: Funciones de activación (sigmoid, relu, tanh, softmax)
    - losses: Funciones de pérdida (MSE, Cross-Entropy)
    - optimizers: Algoritmos de optimización (Adam)
    - trainer: Clase Trainer para gestionar el entrenamiento
    - data_utils: Utilidades para manejo de datos (split, mini-batches)

Ejemplo de uso básico:
    >>> from src.network import NeuralNetwork
    >>> from src.layers import Dense
    >>> from src.optimizers import Adam
    >>> from src.trainer import Trainer
    >>> 
    >>> # Crear red
    >>> net = NeuralNetwork()
    >>> net.add(Dense(784, 128, activation='relu'))
    >>> net.add(Dense(128, 10, activation='softmax'))
    >>> 
    >>> # Entrenar
    >>> optimizer = Adam(lr=0.001)
    >>> trainer = Trainer(net, optimizer, loss_name='cross_entropy')
    >>> trainer.train(X_train, y_train, epochs=50, batch_size=32)

Autores: Raúl Mendoza, Adrián Ojeda, Varela
Asignatura: Optimización y Heurística
Universidad de Las Palmas de Gran Canaria
"""

from .network import NeuralNetwork
from .layers import Dense, Layer
from .activations import sigmoid, relu, tanh, softmax
from .losses import mse, cross_entropy
from .optimizers import Adam
from .trainer import Trainer
from .data_utils import train_val_test_split, create_mini_batches

__version__ = "1.0.0"
__author__ = "Raúl Mendoza, Adrián Ojeda, Varela"
