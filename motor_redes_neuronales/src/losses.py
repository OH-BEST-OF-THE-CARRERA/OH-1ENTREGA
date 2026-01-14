"""
losses.py - Funciones de Pérdida para Redes Neuronales
======================================================

Este módulo implementa las funciones de pérdida (loss functions) utilizadas
para medir el error entre las predicciones de la red y los valores reales.
También incluye los gradientes de cada función, necesarios para backpropagation.

Funciones implementadas:
    - MSE (Mean Squared Error): Para problemas de regresión
    - Cross-Entropy: Para problemas de clasificación multiclase

Cada función de pérdida tiene su correspondiente gradiente (_grad)
necesario para iniciar la retropropagación.

Autores: Raúl Mendoza, Adrián Ojeda, Varela
Asignatura: Optimización y Heurística - ULPGC
"""

import numpy as np


def mse(y_pred, y_true):
    """
    Error Cuadrático Medio (Mean Squared Error).
    
    Función de pérdida estándar para problemas de regresión.
    Penaliza cuadráticamente las diferencias entre predicción y valor real.
    
    Fórmula: MSE = (1/n) * Σ(y_pred - y_true)²
    
    Parameters
    ----------
    y_pred : numpy.ndarray
        Predicciones de la red, forma (batch_size, output_dim).
    y_true : numpy.ndarray
        Valores objetivo reales, misma forma que y_pred.
        
    Returns
    -------
    float
        Valor escalar del error cuadrático medio.
        
    Examples
    --------
    >>> mse(np.array([[1, 2]]), np.array([[1, 3]]))
    0.5
    """
    return np.mean((y_pred - y_true) ** 2)


def mse_grad(y_pred, y_true):
    """
    Gradiente del Error Cuadrático Medio.
    
    Derivada parcial de MSE respecto a y_pred.
    
    Fórmula: ∂MSE/∂y_pred = (2/n) * (y_pred - y_true)
    
    Parameters
    ----------
    y_pred : numpy.ndarray
        Predicciones de la red.
    y_true : numpy.ndarray
        Valores objetivo reales.
        
    Returns
    -------
    numpy.ndarray
        Gradiente con la misma forma que y_pred.
    """
    return 2 * (y_pred - y_true) / y_true.size

def cross_entropy(y_pred, y_true, eps=1e-12):
    """
    Entropía Cruzada Categórica (Categorical Cross-Entropy).
    
    Función de pérdida estándar para clasificación multiclase.
    Mide la divergencia entre la distribución de probabilidad predicha
    y la distribución real (one-hot encoded).
    
    Fórmula: CE = -(1/n) * Σ Σ y_true * log(y_pred)
    
    Se utiliza clipping para evitar log(0) que produciría -inf.
    
    Parameters
    ----------
    y_pred : numpy.ndarray
        Probabilidades predichas, forma (batch_size, n_classes).
        Debe ser salida de softmax (valores entre 0 y 1, suma 1 por fila).
    y_true : numpy.ndarray
        Etiquetas en formato one-hot, forma (batch_size, n_classes).
    eps : float, optional
        Valor pequeño para estabilidad numérica (default: 1e-12).
        
    Returns
    -------
    float
        Valor escalar de la entropía cruzada promedio.
        
    Examples
    --------
    >>> y_true = np.array([[1, 0, 0], [0, 1, 0]])
    >>> y_pred = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1]])
    >>> cross_entropy(y_pred, y_true)
    0.158...
    """
    y_pred = np.clip(y_pred, eps, 1. - eps)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))


def cross_entropy_grad(y_pred, y_true):
    """
    Gradiente de Cross-Entropy combinado con Softmax.
    
    Cuando la capa de salida usa softmax y la pérdida es cross-entropy,
    el gradiente se simplifica elegantemente a (y_pred - y_true).
    
    Esta simplificación es matemáticamente correcta y numéricamente
    más estable que calcular las derivadas por separado.
    
    Fórmula: ∂L/∂z = (y_pred - y_true) / batch_size
    
    donde z son los logits (pre-softmax).
    
    Parameters
    ----------
    y_pred : numpy.ndarray
        Probabilidades predichas (salida de softmax).
    y_true : numpy.ndarray
        Etiquetas en formato one-hot.
        
    Returns
    -------
    numpy.ndarray
        Gradiente con la misma forma que y_pred, normalizado por batch_size.
    """
    batch_size = y_true.shape[0]
    return (y_pred - y_true) / batch_size
