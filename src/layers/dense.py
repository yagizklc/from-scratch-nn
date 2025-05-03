from typing import Optional
from src.base import AbstractLayer

import numpy as np

from src.activation_function import ActivationFunction, ReLU


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

    def forward(self, x: np.ndarray) -> float:
        self.input = x
        self.weighted_sum = np.dot(x, self.weights) + self.bias
        self.output = self.activation(self.weighted_sum)
        return self.output

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
        self.neurons = [
            Perceptron(input_size=input_size, activation=activation)
            for _ in range(output_size)
        ]

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.array([neuron.forward(x=x) for neuron in self.neurons])

    def backward(self, gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        # Ensure input exists before accessing its shape
        assert self.neurons[0].input is not None, (
            "Forward pass must be called before backward pass"
        )

        next_gradient = np.zeros(self.neurons[0].input.shape)

        for i, neuron in enumerate(self.neurons):
            next_gradient += neuron.backward(gradient[i], learning_rate)

        return next_gradient
