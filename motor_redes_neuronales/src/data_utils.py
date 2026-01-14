"""
data_utils.py - Utilidades para Manejo de Datos
================================================

Este módulo proporciona funciones auxiliares para el preprocesamiento
y manejo de datos en el entrenamiento de redes neuronales:
    - División de datos en conjuntos train/validation/test
    - Generación de mini-batches para entrenamiento estocástico
    - Funciones de descarga y preprocesamiento de datasets

Todas las funciones están diseñadas para ser reproducibles mediante
el uso de semillas aleatorias (random seed).

Autores: Raúl Mendoza, Adrián Ojeda, Varela
Asignatura: Optimización y Heurística - ULPGC
"""

import numpy as np


def train_val_test_split(X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42, shuffle=True):
    """
    Divide un dataset en conjuntos de entrenamiento, validación y test.
    
    Permite controlar las proporciones de cada conjunto y garantizar
    reproducibilidad mediante una semilla aleatoria.
    
    Parameters
    ----------
    X : numpy.ndarray
        Matriz de características de forma (n_samples, n_features).
    y : numpy.ndarray
        Vector o matriz de etiquetas de forma (n_samples,) o (n_samples, n_classes).
    train_ratio : float, optional
        Proporción de datos para entrenamiento (default: 0.7 = 70%).
    val_ratio : float, optional
        Proporción de datos para validación (default: 0.15 = 15%).
    test_ratio : float, optional
        Proporción de datos para test (default: 0.15 = 15%).
    seed : int, optional
        Semilla para el generador aleatorio (default: 42).
        Permite reproducibilidad de la división.
    shuffle : bool, optional
        Si True, mezcla los datos antes de dividir (default: True).
        Recomendado para evitar sesgos por orden de los datos.
        
    Returns
    -------
    tuple
        (X_train, y_train, X_val, y_val, X_test, y_test)
        Seis arrays con los datos divididos.
        
    Raises
    ------
    AssertionError
        Si las proporciones no suman 1.0.
        
    Examples
    --------
    >>> X = np.random.randn(100, 10)
    >>> y = np.random.randint(0, 3, 100)
    >>> X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(
    ...     X, y, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42
    ... )
    >>> len(X_train), len(X_val), len(X_test)
    (80, 10, 10)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Las proporciones deben sumar 1"
    
    n = X.shape[0]
    indices = np.arange(n)
    
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
    
    X = X[indices]
    y = y[indices]

    # Calcular índices de corte
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    # Dividir los datos
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def create_mini_batches(X, y, batch_size):
    """
    Generador de mini-batches para entrenamiento estocástico.
    
    Divide los datos en lotes del tamaño especificado. El último batch
    puede ser más pequeño si el número de muestras no es divisible
    exactamente por batch_size.
    
    Parameters
    ----------
    X : numpy.ndarray
        Datos de entrada de forma (n_samples, n_features).
    y : numpy.ndarray
        Etiquetas de forma (n_samples,) o (n_samples, n_classes).
    batch_size : int
        Número de muestras por batch.
        
    Yields
    ------
    tuple (numpy.ndarray, numpy.ndarray)
        (X_batch, y_batch) - Subconjunto de datos para un mini-batch.
        
    Examples
    --------
    >>> X = np.random.randn(100, 10)
    >>> y = np.random.randn(100, 3)
    >>> for X_batch, y_batch in create_mini_batches(X, y, batch_size=32):
    ...     print(f"Batch shape: {X_batch.shape}")
    Batch shape: (32, 10)
    Batch shape: (32, 10)
    Batch shape: (32, 10)
    Batch shape: (4, 10)  # Último batch más pequeño
    
    Notes
    -----
    Esta función es un generador (usa yield), lo que significa que
    los batches se crean bajo demanda, ahorrando memoria.
    """
    n = X.shape[0]
    for start in range(0, n, batch_size):
        end = start + batch_size
        yield X[start:end], y[start:end]


# =============================================================================
# Funciones para carga de datasets
# =============================================================================

def load_mnist(data_dir="data/mnist", normalize=True, flatten=True):
    """
    Descarga y carga el dataset MNIST.
    
    MNIST es un dataset de dígitos manuscritos (0-9) ampliamente utilizado
    como benchmark en machine learning. Contiene 60,000 imágenes de 
    entrenamiento y 10,000 de test, cada una de 28x28 píxeles en escala 
    de grises.
    
    Parameters
    ----------
    data_dir : str, optional
        Directorio donde se guardarán/cargarán los archivos (default: "data/mnist").
    normalize : bool, optional
        Si True, normaliza los píxeles al rango [0, 1] (default: True).
    flatten : bool, optional
        Si True, convierte las imágenes 28x28 a vectores de 784 (default: True).
        
    Returns
    -------
    tuple
        (X_train, y_train, X_test, y_test)
        - X_train: Imágenes de entrenamiento, forma (60000, 784) o (60000, 28, 28)
        - y_train: Etiquetas de entrenamiento, forma (60000,)
        - X_test: Imágenes de test, forma (10000, 784) o (10000, 28, 28)
        - y_test: Etiquetas de test, forma (10000,)
        
    Notes
    -----
    Los archivos se descargan automáticamente de la fuente oficial si no
    existen en el directorio especificado. El dataset ocupa aproximadamente
    11 MB comprimido.
    
    Examples
    --------
    >>> X_train, y_train, X_test, y_test = load_mnist()
    >>> print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    Train: (60000, 784), Test: (10000, 784)
    """
    import os
    import gzip
    import urllib.request
    
    # URLs oficiales de MNIST
    base_url = "http://yann.lecun.com/exdb/mnist/"
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz"
    }
    
    # Crear directorio si no existe
    os.makedirs(data_dir, exist_ok=True)
    
    def download_file(filename):
        """Descarga un archivo si no existe."""
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"Descargando {filename}...")
            url = base_url + filename
            urllib.request.urlretrieve(url, filepath)
            print(f"  Guardado en {filepath}")
        return filepath
    
    def load_images(filepath):
        """Carga imágenes desde archivo idx3-ubyte comprimido."""
        with gzip.open(filepath, 'rb') as f:
            # Leer cabecera: magic number, num imágenes, filas, columnas
            magic = int.from_bytes(f.read(4), 'big')
            num_images = int.from_bytes(f.read(4), 'big')
            rows = int.from_bytes(f.read(4), 'big')
            cols = int.from_bytes(f.read(4), 'big')
            # Leer datos de píxeles
            data = np.frombuffer(f.read(), dtype=np.uint8)
            return data.reshape(num_images, rows, cols)
    
    def load_labels(filepath):
        """Carga etiquetas desde archivo idx1-ubyte comprimido."""
        with gzip.open(filepath, 'rb') as f:
            # Leer cabecera: magic number, num etiquetas
            magic = int.from_bytes(f.read(4), 'big')
            num_labels = int.from_bytes(f.read(4), 'big')
            # Leer etiquetas
            data = np.frombuffer(f.read(), dtype=np.uint8)
            return data
    
    # Descargar archivos si es necesario
    train_images_path = download_file(files["train_images"])
    train_labels_path = download_file(files["train_labels"])
    test_images_path = download_file(files["test_images"])
    test_labels_path = download_file(files["test_labels"])
    
    # Cargar datos
    X_train = load_images(train_images_path)
    y_train = load_labels(train_labels_path)
    X_test = load_images(test_images_path)
    y_test = load_labels(test_labels_path)
    
    # Preprocesamiento
    if normalize:
        X_train = X_train.astype(np.float32) / 255.0
        X_test = X_test.astype(np.float32) / 255.0
    
    if flatten:
        X_train = X_train.reshape(X_train.shape[0], -1)  # (60000, 784)
        X_test = X_test.reshape(X_test.shape[0], -1)      # (10000, 784)
    
    print(f"MNIST cargado: {X_train.shape[0]} train, {X_test.shape[0]} test")
    
    return X_train, y_train, X_test, y_test


def to_one_hot(y, num_classes=None):
    """
    Convierte etiquetas enteras a formato one-hot encoding.
    
    El one-hot encoding representa cada clase como un vector binario
    donde solo la posición correspondiente a la clase es 1.
    
    Parameters
    ----------
    y : numpy.ndarray
        Vector de etiquetas enteras de forma (n_samples,).
    num_classes : int, optional
        Número total de clases. Si es None, se infiere del máximo valor en y.
        
    Returns
    -------
    numpy.ndarray
        Matriz one-hot de forma (n_samples, num_classes).
        
    Examples
    --------
    >>> y = np.array([0, 1, 2, 1])
    >>> to_one_hot(y, num_classes=3)
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.],
           [0., 1., 0.]])
    """
    if num_classes is None:
        num_classes = int(np.max(y)) + 1
    
    one_hot = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    one_hot[np.arange(y.shape[0]), y.astype(int)] = 1.0
    return one_hot


def normalize_features(X, method="zscore"):
    """
    Normaliza las características de los datos.
    
    La normalización es importante para que el entrenamiento converja
    correctamente, especialmente con optimizadores adaptativos.
    
    Parameters
    ----------
    X : numpy.ndarray
        Datos de entrada de forma (n_samples, n_features).
    method : str, optional
        Método de normalización:
        - 'zscore': (X - mean) / std (default)
        - 'minmax': (X - min) / (max - min), rango [0, 1]
        
    Returns
    -------
    numpy.ndarray
        Datos normalizados con la misma forma.
        
    Examples
    --------
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])
    >>> X_norm = normalize_features(X, method='zscore')
    """
    if method == "zscore":
        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True) + 1e-8
        return (X - mean) / std
    elif method == "minmax":
        x_min = X.min(axis=0, keepdims=True)
        x_max = X.max(axis=0, keepdims=True)
        return (X - x_min) / (x_max - x_min + 1e-8)
    else:
        raise ValueError(f"Método de normalización '{method}' no soportado")
