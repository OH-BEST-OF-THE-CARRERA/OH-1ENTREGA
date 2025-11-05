import numpy as np
import matplotlib.pyplot as plt

from . import losses
from .data_utils import create_mini_batches


class Trainer:
    def __init__(self, network, optimizer, loss_name="cross_entropy"):
        self.network = network
        self.optimizer = optimizer
        self.loss_name = loss_name
        if loss_name not in ("cross_entropy", "mse"):
            raise ValueError("loss_name debe ser 'cross_entropy' o 'mse'")

    def _loss_and_grad(self, y_pred, y_true):
        if self.loss_name == "cross_entropy":
            loss = losses.cross_entropy(y_pred, y_true)
            grad = losses.cross_entropy_grad(y_pred, y_true)
            return loss, grad
        else:
            loss = losses.mse(y_pred, y_true)
            grad = losses.mse_grad(y_pred, y_true)
            return loss, grad

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=10, batch_size=32, verbose=True):
        train_losses = []
        val_losses = []

        for epoch in range(1, epochs + 1):
            epoch_losses = []
            # entrenamiento por mini-batches
            for X_batch, y_batch in create_mini_batches(X_train, y_train, batch_size):
                # forward
                y_pred = self.network.forward(X_batch)
                # loss + grad
                loss, grad_loss = self._loss_and_grad(y_pred, y_batch)
                epoch_losses.append(loss)

                # backward
                self.network.backward(grad_loss)

                # update
                params = self.network.params()
                grads = self.network.grads()
                self.optimizer.update(params, grads)

            mean_train_loss = np.mean(epoch_losses)
            train_losses.append(mean_train_loss)

            # validación
            if X_val is not None and y_val is not None:
                y_val_pred = self.network.forward(X_val)
                val_loss, _ = self._loss_and_grad(y_val_pred, y_val)
                val_losses.append(val_loss)
                if verbose:
                    print(f"Epoch {epoch}/{epochs} - loss: {mean_train_loss:.4f} - val_loss: {val_loss:.4f}")
            else:
                if verbose:
                    print(f"Epoch {epoch}/{epochs} - loss: {mean_train_loss:.4f}")

        # gráficas sencillas
        plt.figure()
        plt.plot(train_losses, label="train_loss")
        if X_val is not None and y_val is not None:
            plt.plot(val_losses, label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Curva de pérdida")
        plt.show()

        return train_losses, val_losses
