from abc import ABC, abstractmethod
import numpy as np


class AbstractLayer(ABC):
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        pass
