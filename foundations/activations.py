import numpy as np
from numpy.typing import NDArray


class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        your_answer = 1 / (1 + np.exp(-z))
        return np.round(your_answer, 5)

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.maximum(0.0, z)
