import numpy as np
from .base import BaseLayer


class LinearLayer(BaseLayer):
    """
    Слой, выполняющий линейное преобразование y = x @ W.T + b.
    Параметры:
        parameters[0]: W;
        parameters[1]: b;
    Линейное преобразование выполняется для последней оси тензоров, т.е.
     y[B, ..., out_features] = LinearLayer(in_features, out_feautres)(x[B, ..., in_features].)
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.parameters.append(np.ones((self.out_features, self.in_features)))
        self.parameters.append(np.ones((self.out_features)))
        self.parameters_grads.append(np.zeros((self.out_features, self.in_features)))
        self.parameters_grads.append(np.zeros((self.out_features)))

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        return input @ self.parameters[0].T + self.parameters[1]

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        tmp = output_grad.swapaxes(-1, -2) @ self.input
        self.parameters_grads[0] += np.sum(tmp, tuple(range(tmp.ndim-2)))
        self.parameters_grads[1] += np.sum(output_grad, tuple(range(output_grad.ndim-1)))
        return output_grad @ self.parameters[0]
