"""NumPy port of the JS `MultiLayerNetwork` (see js/network.js).

Same conventions as the original:
  - Fully connected feed-forward
  - ReLU on every hidden layer
  - Linear (identity) activation on the output layer
  - MSE loss, vanilla SGD backprop
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


class MultiLayerNetwork:
    def __init__(self, architecture: Sequence[int], rng: np.random.Generator | None = None):
        if len(architecture) < 2:
            raise ValueError("architecture must have at least input and output layers")
        self.architecture = list(architecture)
        self.layers = len(self.architecture)
        self._rng = rng if rng is not None else np.random.default_rng()
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        for i in range(self.layers - 1):
            self.weights.append(self._rand(self.architecture[i + 1], self.architecture[i]))
            self.biases.append(self._rand(self.architecture[i + 1], 1))
        self.activations: list[np.ndarray] = []
        self.z_values: list[np.ndarray] = []

    def _rand(self, rows: int, cols: int) -> np.ndarray:
        return (self._rng.random((rows, cols)) - 0.5) * 2.0

    def forward(self, x: Sequence[float]) -> np.ndarray:
        """Single-sample forward pass. Mirrors the JS forward(): caches activations and zValues."""
        a = np.asarray(x, dtype=float).reshape(-1, 1)
        self.activations = [a]
        self.z_values = []
        for l in range(self.layers - 1):
            z = self.weights[l] @ self.activations[l] + self.biases[l]
            self.z_values.append(z)
            if l == self.layers - 2:
                a_next = z  # linear output
            else:
                a_next = np.maximum(z, 0.0)
            self.activations.append(a_next)
        return self.activations[-1].ravel()

    def forward_batch(self, X: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        """Vectorized forward pass.

        Args:
            X: shape (n_inputs, n_samples)
        Returns:
            (output, hidden_binary_masks) where output has shape (n_outputs, n_samples)
            and hidden_binary_masks is a list with one (layer_size, n_samples) uint8 array
            per *hidden* layer (matching `getBinaryState()` semantics from the JS version).
        """
        if X.ndim != 2 or X.shape[0] != self.architecture[0]:
            raise ValueError(
                f"X must have shape (input_size={self.architecture[0]}, n_samples); got {X.shape}"
            )
        a = X
        masks: list[np.ndarray] = []
        for l in range(self.layers - 1):
            z = self.weights[l] @ a + self.biases[l]
            if l == self.layers - 2:
                a = z  # linear output, no mask collected
            else:
                mask = (z > 0).astype(np.uint8)
                masks.append(mask)
                a = z * mask
        return a, masks

    def backward(self, x: Sequence[float], target: Sequence[float], lr: float) -> float:
        """Single-sample SGD step with MSE loss. Returns the loss before update."""
        out = self.forward(x)
        target = np.asarray(target, dtype=float)
        err = out - target
        loss = 0.5 * float(err @ err)
        delta = err.reshape(-1, 1)
        for l in range(self.layers - 2, -1, -1):
            dW = delta @ self.activations[l].T
            db = delta
            self.weights[l] -= lr * dW
            self.biases[l] -= lr * db
            if l > 0:
                # Backprop through ReLU of layer l-1 (whose pre-activations live in z_values[l-1]).
                upstream = self.weights[l].T @ delta
                relu_grad = (self.z_values[l - 1] > 0).astype(float)
                delta = upstream * relu_grad
        return loss

    def get_binary_state(self) -> np.ndarray:
        """Concatenated binary activation pattern of the hidden layers (matches JS getBinaryState)."""
        if not self.activations:
            return np.zeros(0, dtype=np.uint8)
        parts = []
        for l in range(1, len(self.activations) - 1):
            parts.append((self.activations[l].ravel() > 0).astype(np.uint8))
        if not parts:
            return np.zeros(0, dtype=np.uint8)
        return np.concatenate(parts)

    @property
    def hidden_neuron_count(self) -> int:
        return sum(self.architecture[1:-1])

    def save_state(self) -> dict:
        return {
            "architecture": list(self.architecture),
            "weights": [w.copy() for w in self.weights],
            "biases": [b.copy() for b in self.biases],
        }

    def load_state(self, state: dict) -> None:
        self.architecture = list(state["architecture"])
        self.layers = len(self.architecture)
        self.weights = [np.array(w, dtype=float) for w in state["weights"]]
        self.biases = [np.array(b, dtype=float) for b in state["biases"]]
        self.activations = []
        self.z_values = []
