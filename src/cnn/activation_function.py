from typing import Callable, Union

import numpy as np
from numpy.typing import NDArray


class ActivationFunction:
    def __init__(
        self,
        function: Callable[[Union[float, NDArray]], Union[float, np.ndarray]],
        derivative: Callable[[Union[float, NDArray]], Union[float, np.ndarray]],
    ) -> None:
        self.function = function
        self.derivative = derivative

    def __call__(self, x: Union[float, NDArray]) -> Union[float, NDArray]:
        return self.function(x)


class ReLU(ActivationFunction):
    """
    ReLU is a function that takes a real number or vector and returns the same.
    The output is the input if it is positive, otherwise it is 0.
    The derivative of ReLU is 1 if the input is positive, otherwise it is 0.
    """

    def __init__(self) -> None:
        super().__init__(lambda x: np.maximum(0, x), lambda x: np.where(x > 0, 1, 0))


class Sigmoid(ActivationFunction):
    """
    Sigmoid is a function that takes a real number or vector and returns values between 0 and 1.
    The output is the sigmoid of the input.
    The derivative of sigmoid is the sigmoid of the input multiplied by 1 minus the sigmoid of the input.
    """

    def __init__(self) -> None:
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        super().__init__(
            sigmoid,
            lambda x: sigmoid(x) * (1 - sigmoid(x)),  # derivative with respect to input
        )


class Softmax(ActivationFunction):
    """
    Softmax is a function that takes a vector of real numbers and returns a vector of the same length.
    The output is a probability distribution over the input values.

    Note: For backpropagation with softmax, the derivative is typically handled
    directly in the loss function (e.g., cross-entropy loss) rather than separately.
    """

    def __init__(self) -> None:
        def softmax(x):
            # Stabilized version to prevent overflow
            exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

        super().__init__(
            softmax,
            lambda x: x
            * (1 - x),  # This is only valid when combined with cross-entropy loss
        )
