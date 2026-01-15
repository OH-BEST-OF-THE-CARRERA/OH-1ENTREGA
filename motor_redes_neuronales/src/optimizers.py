"""
optimizers.py - Algoritmos de Optimización para Redes Neuronales
================================================================

Este módulo implementa los algoritmos de optimización utilizados para
actualizar los parámetros (pesos y sesgos) de la red neuronal durante
el entrenamiento, minimizando la función de pérdida.

Algoritmos implementados:
    - Adam: Adaptive Moment Estimation, combina las ventajas de
            AdaGrad y RMSProp con corrección de sesgo.

El optimizador recibe los parámetros y sus gradientes, y actualiza
los parámetros in-place según las reglas del algoritmo.

Autores: Raúl Mendoza, Adrián Ojeda, Varela
Asignatura: Optimización y Heurística - ULPGC
"""

import numpy as np


class Adam:
    """
    Optimizador Adam (Adaptive Moment Estimation).
    
    Adam combina las ideas de Momentum (momento de primer orden) y
    RMSProp (momento de segundo orden) para adaptar la tasa de
    aprendizaje de cada parámetro individualmente.
    
    Características principales:
        - Adapta el learning rate por parámetro
        - Mantiene promedios móviles de gradientes y gradientes²
        - Incluye corrección de sesgo para las primeras iteraciones
        - Robusto y eficiente para la mayoría de problemas
    
    Algoritmo:
        m_t = β₁ * m_{t-1} + (1 - β₁) * g_t           (momento)
        v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²          (velocidad)
        m̂_t = m_t / (1 - β₁^t)                       (corrección)
        v̂_t = v_t / (1 - β₂^t)                       (corrección)
        θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)        (actualización)
    
    Attributes
    ----------
    lr : float
        Tasa de aprendizaje (α). Default: 0.001
    beta1 : float
        Coeficiente de decaimiento para el primer momento. Default: 0.9
    beta2 : float
        Coeficiente de decaimiento para el segundo momento. Default: 0.999
    eps : float
        Término de estabilidad numérica. Default: 1e-8
    t : int
        Contador de iteraciones para la corrección de sesgo.
    m : dict
        Estimaciones del primer momento (media de gradientes).
    v : dict
        Estimaciones del segundo momento (varianza de gradientes).
    
    References
    ----------
    Kingma, D. P., & Ba, J. (2014). Adam: A Method for Stochastic Optimization.
    arXiv preprint arXiv:1412.6980.
    
    Examples
    --------
    >>> optimizer = Adam(lr=0.001)
    >>> # En el bucle de entrenamiento:
    >>> optimizer.update(network.params(), network.grads())
    """
    
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        """
        Inicializa el optimizador Adam.
        
        Parameters
        ----------
        lr : float, optional
            Tasa de aprendizaje (default: 0.001).
        beta1 : float, optional
            Coeficiente para el primer momento (default: 0.9).
        beta2 : float, optional
            Coeficiente para el segundo momento (default: 0.999).
        eps : float, optional
            Término de estabilidad numérica (default: 1e-8).
        """
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = {}
        self.v = {}

    def update(self, params, grads):
        """
        Actualiza los parámetros usando el algoritmo Adam.
        
        Realiza un paso de optimización: calcula los momentos,
        aplica la corrección de sesgo y actualiza los parámetros in-place.
        
        Parameters
        ----------
        params : list of numpy.ndarray
            Lista de parámetros de la red [W1, b1, W2, b2, ...].
        grads : list of numpy.ndarray
            Lista de gradientes correspondientes a cada parámetro.
            
        Notes
        -----
        Los parámetros se actualizan in-place (modificación directa).
        El contador t se incrementa automáticamente en cada llamada.
        """
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            if i not in self.m:
                self.m[i] = np.zeros_like(g)
                self.v[i] = np.zeros_like(g)

            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
