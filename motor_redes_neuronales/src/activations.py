"""
activations.py - Funciones de Activación para Redes Neuronales
================================================================

Este módulo implementa las funciones de activación más comunes utilizadas
en redes neuronales, junto con sus derivadas para el cálculo del gradiente
durante la retropropagación (backpropagation).

Funciones implementadas:
    - sigmoid: Función logística, salida en rango (0, 1)
    - relu: Rectified Linear Unit, f(x) = max(0, x)
    - tanh: Tangente hiperbólica, salida en rango (-1, 1)
    - softmax: Normalización exponencial para clasificación multiclase

Cada función de activación tiene su correspondiente derivada (_derivative)
necesaria para el algoritmo de backpropagation.

Autores: Raúl Mendoza, Adrián Ojeda, Varela
Asignatura: Optimización y Heurística - ULPGC
"""

import numpy as np


def sigmoid(x):
    """
    Función de activación Sigmoid (logística).
    
    Transforma cualquier valor real al rango (0, 1), lo que la hace útil
    para problemas de clasificación binaria o como activación en capas ocultas.
    
    Fórmula: σ(x) = 1 / (1 + e^(-x))
    
    Parameters
    ----------
    x : numpy.ndarray
        Array de entrada de cualquier forma.
        
    Returns
    -------
    numpy.ndarray
        Array con la misma forma que x, con valores en el rango (0, 1).
        
    Examples
    --------
    >>> sigmoid(np.array([0, 1, -1]))
    array([0.5, 0.73105858, 0.26894142])
    """
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(x):
    """
    Derivada de la función Sigmoid.
    
    Utilizada en backpropagation para calcular el gradiente.
    
    Fórmula: σ'(x) = σ(x) * (1 - σ(x))
    
    Parameters
    ----------
    x : numpy.ndarray
        Array de entrada (valores pre-activación z).
        
    Returns
    -------
    numpy.ndarray
        Derivada de sigmoid evaluada en x.
    """
    s = sigmoid(x)
    return s * (1 - s)


def relu(x):
    """
    Función de activación ReLU (Rectified Linear Unit).
    
    Es la función de activación más utilizada en redes profundas debido
    a su simplicidad computacional y capacidad para mitigar el problema
    del gradiente desvaneciente.
    
    Fórmula: f(x) = max(0, x)
    
    Parameters
    ----------
    x : numpy.ndarray
        Array de entrada de cualquier forma.
        
    Returns
    -------
    numpy.ndarray
        Array con la misma forma, donde los valores negativos son 0.
        
    Examples
    --------
    >>> relu(np.array([-2, -1, 0, 1, 2]))
    array([0, 0, 0, 1, 2])
    """
    return np.maximum(0, x)

def relu_derivative(x):
    """
    Derivada de la función ReLU.
    
    Fórmula: f'(x) = 1 si x > 0, else 0
    
    Parameters
    ----------
    x : numpy.ndarray
        Array de entrada (valores pre-activación z).
        
    Returns
    -------
    numpy.ndarray
        Array binario: 1 donde x > 0, 0 en otro caso.
    """
    return (x > 0).astype(x.dtype)


def tanh(x):
    """
    Función de activación Tangente Hiperbólica.
    
    Similar a sigmoid pero con salida en el rango (-1, 1),
    lo que la hace centrada en cero y puede acelerar la convergencia.
    
    Fórmula: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    
    Parameters
    ----------
    x : numpy.ndarray
        Array de entrada de cualquier forma.
        
    Returns
    -------
    numpy.ndarray
        Array con valores en el rango (-1, 1).
    """
    return np.tanh(x)


def tanh_derivative(x):
    """
    Derivada de la función Tangente Hiperbólica.
    
    Fórmula: tanh'(x) = 1 - tanh²(x)
    
    Parameters
    ----------
    x : numpy.ndarray
        Array de entrada (valores pre-activación z).
        
    Returns
    -------
    numpy.ndarray
        Derivada de tanh evaluada en x.
    """
    return 1 - np.tanh(x) ** 2


def softmax(x):
    """
    Función de activación Softmax.
    
    Convierte un vector de valores reales en una distribución de
    probabilidad. Es la función estándar para la capa de salida
    en problemas de clasificación multiclase.
    
    Fórmula: softmax(x_i) = e^(x_i) / Σ_j e^(x_j)
    
    Se utiliza el truco de estabilidad numérica restando el máximo
    para evitar overflow en la exponencial.
    
    Parameters
    ----------
    x : numpy.ndarray
        Array de forma (batch_size, n_classes) con los logits.
        
    Returns
    -------
    numpy.ndarray
        Array de forma (batch_size, n_classes) con probabilidades
        que suman 1 por cada fila.
        
    Examples
    --------
    >>> softmax(np.array([[1, 2, 3]]))
    array([[0.09003057, 0.24472847, 0.66524096]])
    """
    # x: (batch, n_classes)
    # Truco de estabilidad numérica: restar el máximo
    x_shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)
