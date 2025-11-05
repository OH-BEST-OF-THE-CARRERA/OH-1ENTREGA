import numpy as np

from src.network import NeuralNetwork
from src.layers import Dense
from src.optimizers import Adam
from src.trainer import Trainer

# datos XOR
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
], dtype=float)

y = np.array([
    [1, 0],  # 0
    [0, 1],  # 1
    [0, 1],  # 1
    [1, 0],  # 0
], dtype=float)

# modelo
net = NeuralNetwork()
net.add(Dense(2, 4, activation="tanh"))
net.add(Dense(4, 2, activation="softmax"))

optimizer = Adam(lr=0.05)
trainer = Trainer(net, optimizer, loss_name="cross_entropy")

trainer.train(X, y, epochs=500, batch_size=4, verbose=False)

# evaluación
pred = net.forward(X)
print("Predicciones:")
print(pred)
print("Clase predicha:", np.argmax(pred, axis=1))
print("Clase real    :", np.argmax(y, axis=1))
