from typing import Callable

import numpy as np


class ActivationFunction:
    def __init__(
        self, function: Callable[[float], float], derivative: Callable[[float], float]
    ) -> None:
        self.function = function
        self.derivative = derivative

    def __call__(self, x: float) -> float:
        return self.function(x)


class ReLU(ActivationFunction):
    """
    ReLU is a function that takes a real number and returns a real number.
    The output is the input if it is positive, otherwise it is 0.
    The derivative of ReLU is 1 if the input is positive, otherwise it is 0.
    """

    def __init__(self) -> None:
        super().__init__(lambda x: max(0, x), lambda x: 1 if x > 0 else 0)


class Sigmoid(ActivationFunction):
    """
    Sigmoid is a function that takes a real number and returns a real number between 0 and 1.
    The output is the sigmoid of the input.
    The derivative of sigmoid is the sigmoid of the input multiplied by 1 minus the sigmoid of the input.
    """

    def __init__(self) -> None:
        super().__init__(
            lambda x: 1 / (1 + np.exp(-x)),
            lambda x: x * (1 - x),  # derivative when x is already sigmoid output
        )


class Softmax(ActivationFunction):
    """
    Softmax is a function that takes a vector of real numbers and returns a vector of the same length.
    The output is a probability distribution over the input values.
    The derivative of softmax is the same as the derivative of sigmoid.
    """

    def __init__(self) -> None:
        super().__init__(
            lambda x: np.exp(x) / np.sum(np.exp(x)),
            lambda x: x * (1 - x),  # derivative when x is already softmax output
        )
