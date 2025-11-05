class NeuralNetwork:
    def __init__(self, layers=None):
        self.layers = layers or []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_output):
        # se propaga hacia atrás
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)

    def params(self):
        params = []
        for layer in self.layers:
            params.extend(layer.params())
        return params

    def grads(self):
        grads = []
        for layer in self.layers:
            grads.extend(layer.grads())
        return grads

    def zero_grad(self):
        # opcional: si quisieras poner los grads a cero aquí
        pass
