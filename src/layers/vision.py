from typing import Optional

import numpy as np

from src.activation_function import ActivationFunction, ReLU
from src.base import AbstractLayer


class ConvolutionalLayer(AbstractLayer):
    def __init__(
        self,
        input_shape: tuple[int, int, int],  # (channels, height, width)
        kernel_size: int,
        num_filters: int,
        stride: int = 1,
        padding: int = 0,
        activation: Optional[ActivationFunction] = None,
    ) -> None:
        self.input_channels, self.input_height, self.input_width = input_shape
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.stride = stride
        self.padding = padding
        self.activation = activation or ReLU()

        # Initialize filters/kernels and biases
        self.filters = (
            np.random.randn(num_filters, self.input_channels, kernel_size, kernel_size)
            * 0.1
        )  # Scale to avoid large initial values

        self.biases = np.zeros(num_filters)

        # Cache for backward pass
        self.input = None
        self.output_before_activation = None
        self.output = None

    def _get_output_dimensions(self) -> tuple[int, int]:
        """Calculate output dimensions after convolution."""
        output_height = (
            self.input_height - self.kernel_size + 2 * self.padding
        ) // self.stride + 1
        output_width = (
            self.input_width - self.kernel_size + 2 * self.padding
        ) // self.stride + 1
        return output_height, output_width

    def _pad_input(self, x: np.ndarray) -> np.ndarray:
        """Add padding to input if needed."""
        if self.padding > 0:
            return np.pad(
                x,
                (
                    (0, 0),
                    (0, 0),
                    (self.padding, self.padding),
                    (self.padding, self.padding),
                ),
                mode="constant",
            )
        return x

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass for convolutional layer.

        Args:
            x: Input of shape (batch_size, channels, height, width)

        Returns:
            Output of shape (batch_size, num_filters, output_height, output_width)
        """
        self.input = x
        batch_size = x.shape[0]

        # Apply padding
        x_padded = self._pad_input(x)

        # Calculate output dimensions
        output_height, output_width = self._get_output_dimensions()

        # Initialize output
        self.output_before_activation = np.zeros(
            (batch_size, self.num_filters, output_height, output_width)
        )

        # Perform convolution
        for b in range(batch_size):
            for f in range(self.num_filters):
                for h in range(output_height):
                    h_start = h * self.stride
                    h_end = h_start + self.kernel_size

                    for w in range(output_width):
                        w_start = w * self.stride
                        w_end = w_start + self.kernel_size

                        # Extract the current patch
                        patch = x_padded[b, :, h_start:h_end, w_start:w_end]

                        # Compute convolution
                        self.output_before_activation[b, f, h, w] = (
                            np.sum(patch * self.filters[f]) + self.biases[f]
                        )

        # Apply activation function
        self.output = np.vectorize(self.activation.function)(
            self.output_before_activation
        )
        return self.output

    def backward(self, gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Backward pass for convolutional layer.

        Args:
            gradient: Gradient from next layer of shape (batch_size, num_filters, output_height, output_width)
            learning_rate: Learning rate for parameter updates

        Returns:
            Gradient for previous layer
        """
        # Ensure input and output exist
        assert self.input is not None, (
            "Forward pass must be called before backward pass"
        )
        assert self.output is not None, (
            "Forward pass must be called before backward pass"
        )

        batch_size = self.input.shape[0]

        # Apply activation gradient
        activation_gradient = np.vectorize(self.activation.derivative)(
            self.output_before_activation
        )
        gradient = gradient * activation_gradient

        # Calculate output dimensions
        output_height, output_width = self._get_output_dimensions()

        # Initialize gradients
        filter_gradients = np.zeros_like(self.filters)
        bias_gradients = np.zeros_like(self.biases)

        # Initialize gradient for previous layer
        input_gradient = np.zeros_like(self.input)
        x_padded = self._pad_input(self.input)

        # Calculate gradients
        for b in range(batch_size):
            for f in range(self.num_filters):
                for h in range(output_height):
                    h_start = h * self.stride
                    h_end = h_start + self.kernel_size

                    for w in range(output_width):
                        w_start = w * self.stride
                        w_end = w_start + self.kernel_size

                        # Extract the current patch
                        patch = x_padded[b, :, h_start:h_end, w_start:w_end]

                        # Update filter gradients
                        filter_gradients[f] += patch * gradient[b, f, h, w]

                        # Update bias gradients
                        bias_gradients[f] += gradient[b, f, h, w]

        # Calculate input gradients (for previous layer)
        if self.padding == 0:
            # Simplified case for no padding
            for b in range(batch_size):
                for f in range(self.num_filters):
                    for h in range(output_height):
                        h_start = h * self.stride
                        h_end = h_start + self.kernel_size

                        for w in range(output_width):
                            w_start = w * self.stride
                            w_end = w_start + self.kernel_size

                            # Update input gradients
                            input_gradient[b, :, h_start:h_end, w_start:w_end] += (
                                self.filters[f] * gradient[b, f, h, w]
                            )
        else:
            # Handle padded case
            pad_height = x_padded.shape[2] - self.input.shape[2]
            pad_width = x_padded.shape[3] - self.input.shape[3]
            pad_top = pad_height // 2
            pad_left = pad_width // 2

            for b in range(batch_size):
                for f in range(self.num_filters):
                    for h in range(output_height):
                        h_start = h * self.stride
                        h_end = h_start + self.kernel_size

                        for w in range(output_width):
                            w_start = w * self.stride
                            w_end = w_start + self.kernel_size

                            # Calculate the corresponding region in the input (accounting for padding)
                            in_h_start = max(0, h_start - pad_top)
                            in_h_end = min(self.input.shape[2], h_end - pad_top)
                            in_w_start = max(0, w_start - pad_left)
                            in_w_end = min(self.input.shape[3], w_end - pad_left)

                            # Only update gradient for the actual input (not padding)
                            if in_h_end > in_h_start and in_w_end > in_w_start:
                                # Calculate corresponding filter positions
                                f_h_start = max(0, pad_top - h_start)
                                f_w_start = max(0, pad_left - w_start)

                                input_gradient[
                                    b, :, in_h_start:in_h_end, in_w_start:in_w_end
                                ] += (
                                    self.filters[
                                        f,
                                        :,
                                        f_h_start : f_h_start + (in_h_end - in_h_start),
                                        f_w_start : f_w_start + (in_w_end - in_w_start),
                                    ]
                                    * gradient[b, f, h, w]
                                )

        # Update filters and biases
        self.filters -= learning_rate * filter_gradients / batch_size
        self.biases -= learning_rate * bias_gradients / batch_size

        return input_gradient


class MaxPoolingLayer(AbstractLayer):
    def __init__(
        self,
        input_shape: tuple[int, int, int],  # (channels, height, width)
        pool_size: int,
        stride: Optional[int] = None,
    ) -> None:
        """
        Initialize a max pooling layer.

        Args:
            input_shape: Shape of the input (channels, height, width)
            pool_size: Size of the pooling window
            stride: Stride for the pooling operation (defaults to pool_size if None)
        """
        self.input_channels, self.input_height, self.input_width = input_shape
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size

        # Calculate output dimensions
        self.output_height = (self.input_height - self.pool_size) // self.stride + 1
        self.output_width = (self.input_width - self.pool_size) // self.stride + 1

        # Cache for backward pass
        self.input = None
        self.output = None
        self.max_indices = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass for max pooling layer.

        Args:
            x: Input of shape (batch_size, channels, height, width)

        Returns:
            Output of shape (batch_size, channels, output_height, output_width)
        """
        self.input = x
        batch_size = x.shape[0]

        # Initialize output and max indices
        self.output = np.zeros(
            (batch_size, self.input_channels, self.output_height, self.output_width)
        )
        self.max_indices = np.zeros(
            (batch_size, self.input_channels, self.output_height, self.output_width, 2),
            dtype=int,
        )

        # Perform max pooling
        for b in range(batch_size):
            for c in range(self.input_channels):
                for h in range(self.output_height):
                    h_start = h * self.stride
                    h_end = h_start + self.pool_size

                    for w in range(self.output_width):
                        w_start = w * self.stride
                        w_end = w_start + self.pool_size

                        # Extract the current patch
                        patch = x[b, c, h_start:h_end, w_start:w_end]

                        # Find the maximum value and its index
                        max_val = np.max(patch)
                        max_idx = np.unravel_index(np.argmax(patch), patch.shape)

                        # Store the maximum value and its index
                        self.output[b, c, h, w] = max_val
                        self.max_indices[b, c, h, w] = max_idx

        return self.output

    def backward(self, gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Backward pass for max pooling layer.

        Args:
            gradient: Gradient from next layer of shape (batch_size, channels, output_height, output_width)
            learning_rate: Learning rate (not used for pooling layer, but kept for API consistency)

        Returns:
            Gradient for previous layer
        """
        # Ensure input and output exist
        assert self.input is not None, (
            "Forward pass must be called before backward pass"
        )
        assert self.output is not None, (
            "Forward pass must be called before backward pass"
        )
        assert self.max_indices is not None, (
            "Forward pass must be called before backward pass"
        )

        batch_size = self.input.shape[0]

        # Initialize gradient for previous layer
        input_gradient = np.zeros_like(self.input)

        # Distribute gradient through max pooling layer
        for b in range(batch_size):
            for c in range(self.input_channels):
                for h in range(self.output_height):
                    h_start = h * self.stride

                    for w in range(self.output_width):
                        w_start = w * self.stride

                        # Get the stored indices of the maximum value
                        max_h, max_w = self.max_indices[b, c, h, w]

                        # Add gradient to the position of the maximum value
                        input_gradient[b, c, h_start + max_h, w_start + max_w] += (
                            gradient[b, c, h, w]
                        )

        return input_gradient


class FlattenLayer(AbstractLayer):
    def __init__(self):
        """
        Initialize a flatten layer that transforms multi-dimensional input to 2D.
        Used to connect convolutional/pooling layers to fully-connected layers.
        """
        self.input_shape = None
        self.output_shape = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass for flatten layer.

        Args:
            x: Input of shape (batch_size, channels, height, width)

        Returns:
            Output of shape (batch_size, channels * height * width)
        """
        # Store the input shape for backward pass
        self.input_shape = x.shape
        batch_size = x.shape[0]

        # Flatten the input (keep the batch dimension)
        flattened = x.reshape(batch_size, -1)
        self.output_shape = flattened.shape

        return flattened

    def backward(self, gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Backward pass for flatten layer.

        Args:
            gradient: Gradient from next layer of shape (batch_size, flattened_features)
            learning_rate: Learning rate (not used for flatten layer, but kept for API consistency)

        Returns:
            Gradient reshaped to match the original input dimensions
        """
        # Ensure input shape exists
        assert self.input_shape is not None, (
            "Forward pass must be called before backward pass"
        )

        # Reshape gradient back to the original input shape
        return gradient.reshape(self.input_shape)
