"""
trainer.py - Entrenador de Redes Neuronales
============================================

Este módulo implementa la clase Trainer, que gestiona todo el proceso
de entrenamiento de una red neuronal:
    - Generación de mini-batches
    - Bucle de épocas y batches
    - Cálculo de pérdida y gradientes
    - Actualización de parámetros mediante el optimizador
    - Evaluación en conjunto de validación
    - Visualización de curvas de aprendizaje

El Trainer encapsula la lógica de entrenamiento para mantener
la separación entre la definición de la red y su entrenamiento.

Autores: Raúl Mendoza, Adrián Ojeda, Varela
Asignatura: Optimización y Heurística - ULPGC
"""

import numpy as np
import matplotlib.pyplot as plt

from . import losses
from .data_utils import create_mini_batches


class Trainer:
    """
    Entrenador para redes neuronales.
    
    Gestiona el ciclo completo de entrenamiento incluyendo:
        - Iteración por épocas y mini-batches
        - Forward pass, cálculo de pérdida, backward pass
        - Actualización de parámetros con el optimizador
        - Evaluación periódica en datos de validación
        - Registro y visualización de métricas de entrenamiento
    
    Attributes
    ----------
    network : NeuralNetwork
        La red neuronal a entrenar.
    optimizer : Adam (u otro optimizador)
        El algoritmo de optimización para actualizar parámetros.
    loss_name : str
        Nombre de la función de pérdida ('cross_entropy' o 'mse').
    
    Examples
    --------
    >>> from src.network import NeuralNetwork
    >>> from src.layers import Dense
    >>> from src.optimizers import Adam
    >>> 
    >>> net = NeuralNetwork()
    >>> net.add(Dense(784, 128, activation='relu'))
    >>> net.add(Dense(128, 10, activation='softmax'))
    >>> 
    >>> optimizer = Adam(lr=0.001)
    >>> trainer = Trainer(net, optimizer, loss_name='cross_entropy')
    >>> 
    >>> train_losses, val_losses = trainer.train(
    ...     X_train, y_train,
    ...     X_val=X_val, y_val=y_val,
    ...     epochs=50, batch_size=32
    ... )
    """
    
    def __init__(self, network, optimizer, loss_name="cross_entropy"):
        """
        Inicializa el entrenador.
        
        Parameters
        ----------
        network : NeuralNetwork
            La red neuronal a entrenar.
        optimizer : Adam (u otro optimizador compatible)
            Optimizador con método update(params, grads).
        loss_name : str, optional
            Función de pérdida a utilizar:
            - 'cross_entropy': Para clasificación (con softmax)
            - 'mse': Para regresión
            Default: 'cross_entropy'
            
        Raises
        ------
        ValueError
            Si loss_name no es 'cross_entropy' ni 'mse'.
        """
        self.network = network
        self.optimizer = optimizer
        self.loss_name = loss_name
        if loss_name not in ("cross_entropy", "mse"):
            raise ValueError("loss_name debe ser 'cross_entropy' o 'mse'")

    def _loss_and_grad(self, y_pred, y_true):
        """
        Calcula la pérdida y su gradiente.
        
        Parameters
        ----------
        y_pred : numpy.ndarray
            Predicciones de la red.
        y_true : numpy.ndarray
            Etiquetas reales.
            
        Returns
        -------
        tuple (float, numpy.ndarray)
            - loss: Valor escalar de la pérdida
            - grad: Gradiente de la pérdida respecto a y_pred
        """
        if self.loss_name == "cross_entropy":
            loss = losses.cross_entropy(y_pred, y_true)
            grad = losses.cross_entropy_grad(y_pred, y_true)
            return loss, grad
        else:
            loss = losses.mse(y_pred, y_true)
            grad = losses.mse_grad(y_pred, y_true)
            return loss, grad

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=10, batch_size=32, verbose=True):
        """
        Entrena la red neuronal.
        
        Ejecuta el bucle de entrenamiento completo:
        1. Para cada época:
           a. Dividir datos en mini-batches
           b. Para cada batch: forward -> loss -> backward -> update
           c. Evaluar en validación (si se proporciona)
           d. Registrar métricas
        2. Mostrar gráfica de curvas de pérdida
        
        Parameters
        ----------
        X_train : numpy.ndarray
            Datos de entrenamiento, forma (n_samples, n_features).
        y_train : numpy.ndarray
            Etiquetas de entrenamiento (one-hot para clasificación).
        X_val : numpy.ndarray, optional
            Datos de validación.
        y_val : numpy.ndarray, optional
            Etiquetas de validación.
        epochs : int, optional
            Número de épocas de entrenamiento (default: 10).
        batch_size : int, optional
            Tamaño de los mini-batches (default: 32).
        verbose : bool, optional
            Si True, imprime el progreso por época (default: True).
            
        Returns
        -------
        tuple (list, list)
            - train_losses: Lista de pérdidas promedio por época en train
            - val_losses: Lista de pérdidas por época en validación
            
        Notes
        -----
        La función genera automáticamente una gráfica con las curvas
        de pérdida de entrenamiento y validación.
        """
        train_losses = []
        val_losses = []

        for epoch in range(1, epochs + 1):
            epoch_losses = []
            for X_batch, y_batch in create_mini_batches(X_train, y_train, batch_size):
                y_pred = self.network.forward(X_batch)
                loss, grad_loss = self._loss_and_grad(y_pred, y_batch)
                epoch_losses.append(loss)

                self.network.backward(grad_loss)

                params = self.network.params()
                grads = self.network.grads()
                self.optimizer.update(params, grads)

            mean_train_loss = np.mean(epoch_losses)
            train_losses.append(mean_train_loss)

            if X_val is not None and y_val is not None:
                y_val_pred = self.network.forward(X_val)
                val_loss, _ = self._loss_and_grad(y_val_pred, y_val)
                val_losses.append(val_loss)
                if verbose:
                    print(f"Epoch {epoch}/{epochs} - loss: {mean_train_loss:.4f} - val_loss: {val_loss:.4f}")
            else:
                if verbose:
                    print(f"Epoch {epoch}/{epochs} - loss: {mean_train_loss:.4f}")

        plt.figure()
        plt.plot(train_losses, label="train_loss")
        if X_val is not None and y_val is not None:
            plt.plot(val_losses, label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Curva de pérdida")
        plt.close()

        return train_losses, val_losses
