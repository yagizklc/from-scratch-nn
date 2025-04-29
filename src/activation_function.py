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
    def __init__(self) -> None:
        super().__init__(lambda x: max(0, x), lambda x: 1 if x > 0 else 0)


class Sigmoid(ActivationFunction):
    def __init__(self) -> None:
        super().__init__(
            lambda x: 1 / (1 + np.exp(-x)),
            lambda x: x * (1 - x),  # derivative when x is already sigmoid output
        )


class Softmax(ActivationFunction):
    def __init__(self) -> None:
        super().__init__(lambda x: np.exp(x) / np.sum(np.exp(x)), lambda x: x)
