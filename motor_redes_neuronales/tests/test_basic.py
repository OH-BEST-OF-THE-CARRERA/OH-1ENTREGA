"""
test_basic.py - Tests Unitarios para el Motor de Redes Neuronales
==================================================================

Este módulo contiene tests automáticos que verifican el correcto
funcionamiento de los componentes principales del motor.

Tests incluidos:
    - Test de funciones de activación y sus derivadas
    - Test de funciones de pérdida y sus gradientes
    - Test de forward pass de capas densas
    - Test de backward pass y gradientes
    - Test de optimizador Adam
    - Test de entrenamiento end-to-end (XOR)
    - Test de división de datos
    - Test de mini-batches

Ejecución:
    python -m pytest tests/test_basic.py -v
    
O simplemente:
    python tests/test_basic.py

Autores: Raúl Mendoza, Adrián Ojeda, Varela
Asignatura: Optimización y Heurística - ULPGC
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# Importar módulos a testear
from src.activations import sigmoid, sigmoid_derivative, relu, relu_derivative, tanh, tanh_derivative, softmax
from src.losses import mse, mse_grad, cross_entropy, cross_entropy_grad
from src.layers import Dense
from src.network import NeuralNetwork
from src.optimizers import Adam
from src.trainer import Trainer
from src.data_utils import train_val_test_split, create_mini_batches, to_one_hot


def test_sigmoid():
    """Test función sigmoid y su derivada."""
    print("Testing sigmoid...", end=" ")
    
    # Test valores conocidos
    x = np.array([0.0])
    assert np.isclose(sigmoid(x), 0.5), "sigmoid(0) debe ser 0.5"
    
    # Test límites
    assert sigmoid(np.array([100.0]))[0] > 0.99, "sigmoid(100) debe ser ~1"
    assert sigmoid(np.array([-100.0]))[0] < 0.01, "sigmoid(-100) debe ser ~0"
    
    # Test derivada
    x = np.array([0.0])
    deriv = sigmoid_derivative(x)
    assert np.isclose(deriv, 0.25), "sigmoid'(0) debe ser 0.25"
    
    print("✓")


def test_relu():
    """Test función ReLU y su derivada."""
    print("Testing ReLU...", end=" ")
    
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = relu(x)
    expected = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
    assert np.allclose(y, expected), "ReLU no funciona correctamente"
    
    # Test derivada
    deriv = relu_derivative(x)
    expected_deriv = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
    assert np.allclose(deriv, expected_deriv), "ReLU derivative no funciona"
    
    print("✓")


def test_tanh():
    """Test función tanh y su derivada."""
    print("Testing tanh...", end=" ")
    
    x = np.array([0.0])
    assert np.isclose(tanh(x), 0.0), "tanh(0) debe ser 0"
    
    # Test derivada en 0
    deriv = tanh_derivative(x)
    assert np.isclose(deriv, 1.0), "tanh'(0) debe ser 1"
    
    print("✓")


def test_softmax():
    """Test función softmax."""
    print("Testing softmax...", end=" ")
    
    x = np.array([[1.0, 2.0, 3.0]])
    y = softmax(x)
    
    # Softmax debe sumar 1
    assert np.isclose(y.sum(), 1.0), "Softmax debe sumar 1"
    
    # Todos los valores deben ser positivos
    assert np.all(y > 0), "Softmax debe producir valores positivos"
    
    # El valor más alto debe corresponder al input más alto
    assert np.argmax(y) == 2, "Softmax debe preservar el orden"
    
    print("✓")


def test_mse():
    """Test función MSE y su gradiente."""
    print("Testing MSE...", end=" ")
    
    y_pred = np.array([[1.0, 2.0]])
    y_true = np.array([[1.0, 3.0]])
    
    loss = mse(y_pred, y_true)
    assert np.isclose(loss, 0.5), f"MSE incorrecto: {loss}"
    
    # Test gradiente: 2*(y_pred - y_true) / n_elements
    grad = mse_grad(y_pred, y_true)
    expected_grad = 2 * (y_pred - y_true) / y_true.size  # 2*[[0, -1]]/2 = [[0, -1]]
    assert np.allclose(grad, expected_grad), f"Gradiente MSE incorrecto: {grad} vs {expected_grad}"
    
    print("✓")


def test_cross_entropy():
    """Test función Cross-Entropy y su gradiente."""
    print("Testing Cross-Entropy...", end=" ")
    
    y_pred = np.array([[0.9, 0.1]])
    y_true = np.array([[1.0, 0.0]])
    
    loss = cross_entropy(y_pred, y_true)
    expected_loss = -np.log(0.9)
    assert np.isclose(loss, expected_loss, rtol=1e-5), "Cross-entropy incorrecto"
    
    print("✓")


def test_dense_forward():
    """Test forward pass de capa densa."""
    print("Testing Dense forward...", end=" ")
    
    np.random.seed(42)
    layer = Dense(n_in=3, n_out=2, activation=None)
    
    x = np.array([[1.0, 2.0, 3.0]])
    y = layer.forward(x)
    
    # Verificar forma de salida
    assert y.shape == (1, 2), f"Forma incorrecta: {y.shape}"
    
    # Verificar que se puede calcular manualmente
    expected = x @ layer.W + layer.b
    assert np.allclose(y, expected), "Forward pass incorrecto"
    
    print("✓")


def test_dense_backward():
    """Test backward pass de capa densa."""
    print("Testing Dense backward...", end=" ")
    
    np.random.seed(42)
    layer = Dense(n_in=3, n_out=2, activation=None)
    
    x = np.array([[1.0, 2.0, 3.0]])
    y = layer.forward(x)
    
    # Simular gradiente de la pérdida
    grad_output = np.array([[1.0, 1.0]])
    grad_input = layer.backward(grad_output)
    
    # Verificar formas
    assert grad_input.shape == x.shape, "Forma de gradiente incorrecta"
    assert layer.dW.shape == layer.W.shape, "Forma de dW incorrecta"
    assert layer.db.shape == layer.b.shape, "Forma de db incorrecta"
    
    print("✓")


def test_network_forward():
    """Test forward pass de red completa."""
    print("Testing NeuralNetwork forward...", end=" ")
    
    np.random.seed(42)
    net = NeuralNetwork()
    net.add(Dense(2, 4, activation="relu"))
    net.add(Dense(4, 2, activation="softmax"))
    
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = net.forward(x)
    
    # Verificar forma
    assert y.shape == (2, 2), f"Forma incorrecta: {y.shape}"
    
    # Verificar que softmax produce distribución válida
    assert np.allclose(y.sum(axis=1), 1.0), "Softmax no suma 1"
    
    print("✓")


def test_network_backward():
    """Test backward pass de red completa."""
    print("Testing NeuralNetwork backward...", end=" ")
    
    np.random.seed(42)
    net = NeuralNetwork()
    net.add(Dense(2, 4, activation="relu"))
    net.add(Dense(4, 2, activation="softmax"))
    
    x = np.array([[1.0, 2.0]])
    y = net.forward(x)
    
    # Simular gradiente de cross-entropy
    y_true = np.array([[1.0, 0.0]])
    grad = cross_entropy_grad(y, y_true)
    
    net.backward(grad)
    
    # Verificar que los gradientes se calcularon
    grads = net.grads()
    assert len(grads) == 4, "Deben haber 4 gradientes (dW1, db1, dW2, db2)"
    
    for g in grads:
        assert not np.all(g == 0), "Los gradientes no deben ser todos cero"
    
    print("✓")


def test_adam_optimizer():
    """Test optimizador Adam."""
    print("Testing Adam optimizer...", end=" ")
    
    np.random.seed(42)
    optimizer = Adam(lr=0.1)
    
    # Crear un parámetro simple
    param = np.array([1.0, 2.0, 3.0])
    grad = np.array([0.1, 0.2, 0.3])
    
    params = [param.copy()]
    grads = [grad]
    
    # Primer paso
    optimizer.update(params, grads)
    
    # El parámetro debe haber cambiado
    assert not np.allclose(params[0], param), "Adam no actualizó el parámetro"
    
    print("✓")


def test_train_val_test_split():
    """Test división de datos."""
    print("Testing train_val_test_split...", end=" ")
    
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 3, 100)
    
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(
        X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42
    )
    
    # Verificar tamaños
    assert len(X_train) == 70, f"Train debe tener 70 muestras, tiene {len(X_train)}"
    assert len(X_val) == 15, f"Val debe tener 15 muestras, tiene {len(X_val)}"
    assert len(X_test) == 15, f"Test debe tener 15 muestras, tiene {len(X_test)}"
    
    # Verificar que no hay solapamiento
    total = len(X_train) + len(X_val) + len(X_test)
    assert total == 100, "La suma debe ser 100"
    
    print("✓")


def test_mini_batches():
    """Test generador de mini-batches."""
    print("Testing create_mini_batches...", end=" ")
    
    X = np.arange(100).reshape(100, 1)
    y = np.arange(100).reshape(100, 1)
    
    batches = list(create_mini_batches(X, y, batch_size=32))
    
    # Deben haber 4 batches (32 + 32 + 32 + 4)
    assert len(batches) == 4, f"Deben haber 4 batches, hay {len(batches)}"
    
    # Verificar tamaños
    assert batches[0][0].shape[0] == 32, "Primer batch debe tener 32"
    assert batches[-1][0].shape[0] == 4, "Último batch debe tener 4"
    
    print("✓")


def test_to_one_hot():
    """Test conversión a one-hot."""
    print("Testing to_one_hot...", end=" ")
    
    y = np.array([0, 1, 2, 1, 0])
    one_hot = to_one_hot(y, num_classes=3)
    
    expected = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0]
    ], dtype=np.float32)
    
    assert np.allclose(one_hot, expected), "One-hot incorrecto"
    
    print("✓")


def test_xor_training():
    """Test entrenamiento end-to-end con XOR."""
    print("Testing XOR training...", end=" ")
    
    np.random.seed(42)
    
    # Datos XOR
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([[1, 0], [0, 1], [0, 1], [1, 0]], dtype=np.float32)
    
    # Crear red
    net = NeuralNetwork()
    net.add(Dense(2, 8, activation="tanh"))
    net.add(Dense(8, 2, activation="softmax"))
    
    # Entrenar
    optimizer = Adam(lr=0.1)
    trainer = Trainer(net, optimizer, loss_name="cross_entropy")
    train_losses, _ = trainer.train(X, y, epochs=500, batch_size=4, verbose=False)
    
    # Verificar convergencia
    assert train_losses[-1] < train_losses[0], "La pérdida debe disminuir"
    assert train_losses[-1] < 0.1, f"La pérdida final debe ser baja, es {train_losses[-1]:.4f}"
    
    # Verificar predicciones
    pred = net.forward(X)
    pred_labels = np.argmax(pred, axis=1)
    true_labels = np.argmax(y, axis=1)
    accuracy = np.mean(pred_labels == true_labels)
    
    assert accuracy == 1.0, f"XOR debe clasificarse al 100%, accuracy={accuracy}"
    
    print("✓")


def run_all_tests():
    """Ejecuta todos los tests."""
    print("\n" + "="*60)
    print("EJECUTANDO TESTS DEL MOTOR DE REDES NEURONALES")
    print("="*60 + "\n")
    
    tests = [
        test_sigmoid,
        test_relu,
        test_tanh,
        test_softmax,
        test_mse,
        test_cross_entropy,
        test_dense_forward,
        test_dense_backward,
        test_network_forward,
        test_network_backward,
        test_adam_optimizer,
        test_train_val_test_split,
        test_mini_batches,
        test_to_one_hot,
        test_xor_training,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ - {e}")
            failed += 1
        except Exception as e:
            print(f"✗ - Error inesperado: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"RESULTADOS: {passed} passed, {failed} failed")
    print("="*60)
    
    if failed == 0:
        print("\n✅ Todos los tests pasaron correctamente!")
    else:
        print(f"\n❌ {failed} tests fallaron.")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
