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

    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        # x: 1D input array
        # w: 1D weight array (same length as x)
        # b: scalar bias
        # activation: "sigmoid" or "relu"
        #
        # Pre-activation: z = dot(x, w) + b
        # Sigmoid: σ(z) = 1 / (1 + exp(-z))
        # ReLU: max(0, z)
        # return round(your_answer, 5)
        your_answer = np.sum(x * w) + b

        your_answer = self.sigmoid(your_answer) if activation == "sigmoid" else self.relu(your_answer)
            
        return round(your_answer, 5)
