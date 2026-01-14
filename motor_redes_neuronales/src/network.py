"""
network.py - Clase Principal de la Red Neuronal
================================================

Este módulo contiene la clase NeuralNetwork, que representa una
red neuronal densa (Fully Connected Neural Network / Feedforward NN).

La clase gestiona:
    - La arquitectura de capas
    - El cálculo del forward pass (propagación hacia adelante)
    - El cálculo del backward pass (retropropagación de gradientes)
    - El acceso a parámetros y gradientes para el optimizador

La red se construye añadiendo capas secuencialmente con el método add(),
o pasando una lista de capas en el constructor.

Autores: Raúl Mendoza, Adrián Ojeda, Varela
Asignatura: Optimización y Heurística - ULPGC
"""


class NeuralNetwork:
    """
    Red Neuronal Densa (Fully Connected / Feedforward Neural Network).
    
    Implementa una red neuronal multicapa donde cada capa está conectada
    completamente con la siguiente. Soporta un número arbitrario de capas
    y neuronas por capa.
    
    La red implementa:
        - Forward Pass: Propagación de la entrada a través de todas las capas
        - Backward Pass: Retropropagación del gradiente de la pérdida
        - Acceso a parámetros: Para actualización por el optimizador
    
    Attributes
    ----------
    layers : list
        Lista de capas (objetos Layer/Dense) que componen la red.
    
    Examples
    --------
    >>> from src.layers import Dense
    >>> net = NeuralNetwork()
    >>> net.add(Dense(784, 128, activation='relu'))
    >>> net.add(Dense(128, 64, activation='relu'))
    >>> net.add(Dense(64, 10, activation='softmax'))
    >>> output = net.forward(X)  # X de forma (batch, 784)
    """
    
    def __init__(self, layers=None):
        """
        Inicializa la red neuronal.
        
        Parameters
        ----------
        layers : list, optional
            Lista de capas para inicializar la red (default: None).
            Si es None, se crea una red vacía a la que se pueden
            añadir capas con el método add().
        """
        self.layers = layers or []

    def add(self, layer):
        """
        Añade una capa a la red.
        
        Las capas se añaden secuencialmente y se ejecutarán en orden
        durante el forward pass.
        
        Parameters
        ----------
        layer : Layer
            Objeto de capa (Dense) a añadir a la red.
        """
        self.layers.append(layer)

    def forward(self, x):
        """
        Realiza el Forward Pass (propagación hacia adelante).
        
        Pasa la entrada secuencialmente a través de todas las capas,
        donde cada capa aplica su transformación lineal y función
        de activación.
        
        Parameters
        ----------
        x : numpy.ndarray
            Entrada de forma (batch_size, n_features).
            
        Returns
        -------
        numpy.ndarray
            Salida de la red de forma (batch_size, n_outputs).
            
        Notes
        -----
        Durante el forward pass, cada capa almacena los valores
        intermedios necesarios para el backward pass.
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_output):
        """
        Realiza el Backward Pass (retropropagación).
        
        Propaga el gradiente de la función de pérdida hacia atrás
        a través de todas las capas, calculando los gradientes
        de los pesos y sesgos de cada capa.
        
        Parameters
        ----------
        grad_output : numpy.ndarray
            Gradiente de la pérdida respecto a la salida de la red,
            de forma (batch_size, n_outputs). Normalmente viene de
            la función loss_grad().
            
        Notes
        -----
        Los gradientes calculados se almacenan en cada capa (dW, db)
        y se pueden obtener con el método grads().
        """
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)

    def params(self):
        """
        Obtiene todos los parámetros entrenables de la red.
        
        Recorre todas las capas y recopila sus parámetros (pesos W
        y sesgos b) en una lista plana.
        
        Returns
        -------
        list of numpy.ndarray
            Lista con todos los parámetros: [W1, b1, W2, b2, ...]
            
        Notes
        -----
        Los parámetros se devuelven por referencia, por lo que
        las modificaciones del optimizador afectan directamente
        a los parámetros de las capas.
        """
        params = []
        for layer in self.layers:
            params.extend(layer.params())
        return params

    def grads(self):
        """
        Obtiene todos los gradientes calculados de la red.
        
        Recorre todas las capas y recopila los gradientes (dW, db)
        calculados durante el último backward pass.
        
        Returns
        -------
        list of numpy.ndarray
            Lista con todos los gradientes: [dW1, db1, dW2, db2, ...]
            
        Notes
        -----
        Los gradientes corresponden a los parámetros en el mismo orden
        que devuelve params(), permitiendo actualizaciones como:
        optimizer.update(network.params(), network.grads())
        """
        grads = []
        for layer in self.layers:
            grads.extend(layer.grads())
        return grads

    def zero_grad(self):
        """
        Reinicia los gradientes a cero (opcional).
        
        Este método es útil si se quieren acumular gradientes
        de múltiples batches antes de actualizar, o para limpiar
        el estado entre diferentes modos de operación.
        
        Notes
        -----
        En la implementación actual, los gradientes se sobrescriben
        en cada backward pass, por lo que este método es opcional.
        """
        pass
