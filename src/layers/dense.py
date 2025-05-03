from typing import Optional

import numpy as np

from src.activation_function import ActivationFunction, ReLU
from src.base import AbstractLayer


class Perceptron:
    def __init__(
        self, input_size: int, activation: Optional[ActivationFunction] = None
    ) -> None:
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()
        self.activation = activation or ReLU()
        self.input = None
        self.weighted_sum = None
        self.output = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        self.weighted_sum = np.dot(x, self.weights) + self.bias
        self.output = self.activation(self.weighted_sum)
        return np.asarray(self.output)

    def backward(self, gradient: float, learning_rate: float) -> np.ndarray:
        # Gradient of activation function
        assert self.output is not None, "Output is not set"
        activation_gradient = self.activation.derivative(self.output)

        # Calculate gradient for this neuron
        delta = gradient * activation_gradient

        # Calculate gradients for weights and bias
        assert self.input is not None, "Input is not set"
        weight_gradients = delta * self.input
        bias_gradient = delta

        # Update weights and bias
        self.weights -= learning_rate * weight_gradients
        self.bias -= learning_rate * bias_gradient

        # Return gradient for previous layer
        return self.weights * delta


class PerceptronLayer(AbstractLayer):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation: Optional[ActivationFunction] = None,
    ) -> None:
        self.weights = np.random.randn(input_size, output_size)
        self.biases = np.random.randn(output_size)
        self.activation = activation or ReLU()
        self.input = None
        self.output = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Store input for backward pass
        self.input = x

        # Calculate weighted sum: (batch_size, output_size)
        z = np.dot(x, self.weights) + self.biases

        # Apply activation function
        a = np.asarray(self.activation(z))
        self.output = a

        return a

    def backward(self, gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        batch_size = gradient.shape[0]

        # Get activation gradient
        assert self.output is not None, "Forward pass must be called before backward"
        activation_gradient = self.activation.derivative(self.output)
        delta = gradient * activation_gradient

        # Get input gradients
        input_gradient = np.dot(delta, self.weights.T)

        # Get weight gradients
        assert self.input is not None, "Forward pass must be called before backward"
        weight_gradients = np.zeros_like(self.weights)
        for i in range(batch_size):
            weight_gradients += np.outer(self.input[i], delta[i])

        # Get bias gradients
        bias_gradients = np.sum(delta, axis=0)

        # Update weights and biases
        self.weights -= learning_rate * weight_gradients / batch_size
        self.biases -= learning_rate * bias_gradients / batch_size

        return input_gradient
