# test_iris.py
import os
import urllib.request
import numpy as np
import matplotlib.pyplot as plt

from src.network import NeuralNetwork
from src.layers import Dense
from src.optimizers import Adam
from src.trainer import Trainer
from src.data_utils import train_val_test_split


# =========================================================
# 1. Cargar dataset IRIS
# =========================================================

def download_iris(csv_path):
    """
    Descarga el iris de la UCI si no existe.
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    print(f"Descargando IRIS desde {url} ...")
    urllib.request.urlretrieve(url, csv_path)
    print("Descarga completada.")


def load_iris_local(csv_path="data/iris.csv"):
    """
    Carga iris de un csv con el formato UCI:
    sepal_length,sepal_width,petal_length,petal_width,class
    """
    if not os.path.exists("data"):
        os.makedirs("data", exist_ok=True)

    if not os.path.exists(csv_path):
        # intentamos descargar
        download_iris(csv_path)

    X = []
    y = []
    with open(csv_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            # 4 features + label
            features = list(map(float, parts[:4]))
            label = parts[4]
            X.append(features)
            y.append(label)

    X = np.array(X, dtype=float)

    # mapear etiquetas a enteros
    classes = sorted(list(set(y)))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([class_to_idx[c] for c in y], dtype=int)

    return X, y_idx, classes


# =========================================================
# 2. One-hot y normalización
# =========================================================

def to_one_hot(y, num_classes):
    oh = np.zeros((y.shape[0], num_classes))
    oh[np.arange(y.shape[0]), y] = 1.0
    return oh


def normalize_features(X):
    # normalización simple: (x - media)/std
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-8
    return (X - mean) / std


# =========================================================
# 3. main de experimento
# =========================================================

def main():
    # 1) cargar datos
    X, y_idx, classes = load_iris_local()
    X = normalize_features(X)
    y_oh = to_one_hot(y_idx, num_classes=len(classes))

    # 2) split train/val/test
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(
        X, y_oh,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
        shuffle=True
    )

    # 3) definir red
    # iris: 4 features -> queremos 3 clases
    net = NeuralNetwork()
    net.add(Dense(4, 16, activation="relu"))
    net.add(Dense(16, 3, activation="softmax"))

    # 4) entrenador
    optimizer = Adam(lr=0.01)
    trainer = Trainer(net, optimizer, loss_name="cross_entropy")

    # 5) entrenar
    train_losses, val_losses = trainer.train(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        epochs=120,
        batch_size=16,
        verbose=True
    )

    # 6) evaluar en test
    y_test_pred = net.forward(X_test)
    # accuracy
    y_test_labels = np.argmax(y_test, axis=1)
    y_pred_labels = np.argmax(y_test_pred, axis=1)
    accuracy = np.mean(y_test_labels == y_pred_labels)

    print("\n========== RESULTADOS IRIS ==========")
    print(f"Clases: {classes}")
    print(f"Accuracy en test: {accuracy * 100:.2f} %")

    # 7) (opcional) mostrar matriz de confusión sencilla
    num_classes = len(classes)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_test_labels, y_pred_labels):
        cm[t, p] += 1

    print("\nMatriz de confusión (filas = real, columnas = pred):")
    print(cm)

    # gráfica de pérdidas ya la hace el trainer, pero si quieres aparte:
    # plt.figure()
    # plt.plot(train_losses, label="train")
    # plt.plot(val_losses, label="val")
    # plt.legend()
    # plt.show()


if __name__ == "__main__":
    main()

