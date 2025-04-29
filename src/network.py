from typing import Optional

import numpy as np

from src.activation_function import ActivationFunction, ReLU
from src.vision import ConvolutionalLayer, MaxPoolingLayer
from src.loss_function import MSE, LossFunction


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


class Layer:
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
        assert (
            self.neurons[0].input is not None
        ), "Forward pass must be called before backward pass"

        next_gradient = np.zeros(self.neurons[0].input.shape)

        for i, neuron in enumerate(self.neurons):
            next_gradient += neuron.backward(gradient[i], learning_rate)

        return next_gradient


class NeuralNetwork:
    def __init__(self, layers: list[Layer | ConvolutionalLayer | MaxPoolingLayer]) -> None:
        self.layers = layers

    def add_layer(
        self,
        input_size: int,
        output_size: int,
        activation: Optional[ActivationFunction] = None,
    ) -> None:
        self.layers.append(
            Layer(
                input_size=input_size,
                output_size=output_size,
                activation=activation,
            )
        )

    def add_conv_layer(
        self,
        input_shape: tuple[int, int, int],  # (channels, height, width)
        kernel_size: int,
        num_filters: int,
        stride: int = 1,
        padding: int = 0,
        activation: Optional[ActivationFunction] = None,
    ) -> None:
        """
        Add a convolutional layer to the network.

        Args:
            input_shape: Shape of the input (channels, height, width)
            kernel_size: Size of the convolutional kernel (square)
            num_filters: Number of filters to apply
            stride: Stride for the convolution operation
            padding: Padding to apply to the input
            activation: Activation function to use
        """
        self.layers.append(
            ConvolutionalLayer(
                input_shape=input_shape,
                kernel_size=kernel_size,
                num_filters=num_filters,
                stride=stride,
                padding=padding,
                activation=activation,
            )
        )

    def add_max_pooling_layer(
        self,
        input_shape: tuple[int, int, int],  # (channels, height, width)
        pool_size: int,
        stride: Optional[int] = None,
    ) -> None:
        """
        Add a max pooling layer to the network.

        Args:
            input_shape: Shape of the input (channels, height, width)
            pool_size: Size of the pooling window
            stride: Stride for the pooling operation (defaults to pool_size if None)
        """
        self.layers.append(
            MaxPoolingLayer(
                input_shape=input_shape,
                pool_size=pool_size,
                stride=stride,
            )
        )

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

    def train(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int = 1000,
        learning_rate: float = 0.01,
        loss_function: LossFunction = MSE(),
        batch_size: Optional[int] = None,
    ) -> list[float]:
        """
        Train the neural network.

        Args:
            x: Input data (for convolutional layers, shape should be (batch_size, channels, height, width))
            y: Target data
            epochs: Number of training epochs
            learning_rate: Learning rate for parameter updates
            loss_function: Loss function to use
            batch_size: Size of mini-batches (if None, use all data)

        Returns:
            List of loss values for each epoch
        """
        num_samples = x.shape[0]
        losses = []

        # Use full batch if batch_size is None
        batch_size = num_samples if batch_size is None else batch_size

        for epoch in range(epochs):
            epoch_loss = 0.0

            # Shuffle data for each epoch
            indices = np.random.permutation(num_samples)
            x_shuffled = x[indices]
            y_shuffled = y[indices]

            # Mini-batch training
            for i in range(0, num_samples, batch_size):
                x_batch = x_shuffled[i : i + batch_size]
                y_batch = y_shuffled[i : i + batch_size]

                # Forward pass
                y_pred = self.forward(x_batch)

                # Calculate loss
                batch_loss = loss_function.function(y_batch, y_pred)
                epoch_loss += batch_loss * len(x_batch) / num_samples

                # Calculate gradients
                gradient = loss_function.derivative(y_batch, y_pred)

                # Backward pass
                self.backward(gradient, learning_rate)

            losses.append(epoch_loss)

            # Optional: Print progress
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {epoch_loss:.4f}")

        return losses
