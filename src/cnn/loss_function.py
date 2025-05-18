from typing import Callable

import numpy as np


class LossFunction:
    def __init__(
        self,
        function: Callable[[np.ndarray, np.ndarray], float],
        derivative: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ) -> None:
        self.function = function
        self.derivative = derivative


class MSE(LossFunction):
    def __init__(self) -> None:
        super().__init__(
            lambda y_true, y_pred: float(np.mean(np.square(y_true - y_pred))),
            lambda y_true, y_pred: 2 * (y_pred - y_true) / y_true.size,
        )
