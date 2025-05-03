
import numpy as np

from src.base import AbstractLayer



class NeuralNetwork:
    def __init__(self, layers: list[AbstractLayer]) -> None:
        self.layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        output = x
        for layer in self.layers:
            output = layer.forward(x=output)
        return output

    def backward(self, gradient: np.ndarray, learning_rate: float) -> None:
        """
        Backward pass through all layers.

        Args:
            gradient: Gradient from loss function
            learning_rate: Learning rate for parameter updates
        """
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient=gradient, learning_rate=learning_rate)
