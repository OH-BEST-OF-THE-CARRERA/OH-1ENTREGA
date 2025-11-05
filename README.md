# Motor de Redes Neuronales desde Cero

Proyecto desarrollado para la asignatura **Optimización y Heurística**  
Grado en Ciencia e Ingeniería de Datos – ULPGC

Autores: **Raúl Mendoza**, **Adrián Ojdeva** y **Varela**  
Curso 2025

---

## Descripción General

Este proyecto implementa un **motor de redes neuronales densas (Fully Connected Neural Networks)** completamente desde cero, utilizando únicamente **NumPy** y **matplotlib**, sin frameworks de deep learning como TensorFlow o PyTorch.

El objetivo es comprender a fondo el funcionamiento del **Forward Pass**, el **Backpropagation** y los algoritmos de optimización como **Adam**, demostrando aprendizaje real sobre datasets clásicos como **XOR** e **IRIS**.

---

## Características Principales

- Arquitecturas configurables (número de capas y neuronas)
- Funciones de activación: `sigmoid`, `tanh`, `ReLU`, `softmax`
- Forward Pass y Backpropagation implementados manualmente
- Funciones de pérdida: `MSE`, `Cross-Entropy`
- Optimizador **Adam** encapsulado en clase independiente
- Entrenamiento por **mini-batches** con tamaño variable
- División automática en **train / validation / test**
- Gráficas de pérdida y métricas de rendimiento
- Modularización total del código (`src/`)

---

## Estructura del Proyecto

```
motor_redes_neuronales/
│
├── src/
│   ├── activations.py
│   ├── layers.py
│   ├── losses.py
│   ├── optimizers.py
│   ├── network.py
│   ├── trainer.py
│   └── data_utils.py
│
├── notebooks/
│   ├── test_xor.ipynb
│   ├── test_iris.ipynb
│   └── test_mnist.ipynb
│
├── tests/
│   └── test_basic.py
│
├── data/
│   └── iris.csv
│
├── xor_example.py
├── test_iris.py
├── memoria/
│   ├── memoria_motor_redes_neuronales.tex
│   └── fig_loss_iris.png
│
├── README.md
└── .gitignore
```

---

## Ejecución del Proyecto

### 1️ Crear entorno virtual

```bash
python -m venv venv
venv\Scripts\activate    # En Windows
pip install numpy matplotlib
```

### 2️ Ejecutar el experimento XOR

```bash
python xor_example.py
```

### 3️ Ejecutar el experimento IRIS

```bash
python test_iris.py
```

---

## Resultados

| Experimento | Dataset | Arquitectura | Accuracy | Observaciones |
|--------------|----------|---------------|-----------|----------------|
| XOR | 2–4–2 | tanh + softmax | 100 % | Aprendizaje perfecto |
| IRIS | 4–16–3 | ReLU + softmax | 91.3 % | Aprendizaje estable |

**Figura 1.** Curva de pérdida del entrenamiento sobre IRIS  
![Curva de pérdida](memoria/fig_loss_iris.png)

---

## Implementación Técnica

- **Capa (`Dense`)**: cálculo \( z = Wx + b \) y derivadas.
- **Backpropagation**: regla de la cadena implementada manualmente.
- **Optimizador (`Adam`)**: actualización adaptativa de parámetros.
- **Trainer**: controla los mini-batches, las épocas y la validación.

---

## Memoria del Proyecto

El informe completo en formato paper se encuentra en:  
[`memoria/memoria_motor_redes_neuronales.tex`](memoria/memoria_motor_redes_neuronales.tex)

---

## Trabajo Futuro

- Añadir nuevas funciones de activación (`LeakyReLU`, `ELU`)
- Regularización (L2, Dropout)
- Nuevos optimizadores (`SGD+Momentum`, `RMSProp`)
- Datasets adicionales (Fashion-MNIST, CIFAR-10)
- Interfaz gráfica para visualización de entrenamiento

---

## Licencia

Proyecto académico desarrollado para la ULPGC.  
Uso educativo permitido citando la fuente.

---

**Autores:**  
Raúl Mendoza — Adrián Ojdeva — Varela  
*Optimización y Heurística – ULPGC, 2025*
