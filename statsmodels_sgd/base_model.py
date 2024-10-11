import torch.nn as nn
import torch.optim as optim
import numpy as np
from .tools import (
    add_constant,
    calculate_standard_errors,
    calculate_t_p_values,
)

import sys

print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Python path: {sys.path}")

try:
    import torch

    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch installation path: {torch.__file__}")
except ImportError as e:
    print(f"Error importing torch: {e}")
    print("Installed packages:")
    import subprocess

    result = subprocess.run(
        [sys.executable, "-m", "pip", "list"], capture_output=True, text=True
    )
    print(result.stdout)


class BaseModel(nn.Module):
    def __init__(
        self,
        n_features,
        learning_rate=0.01,
        epochs=1000,
        batch_size=32,
        clip_value=1.0,
    ):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.clip_value = clip_value
        self.results_ = None

    def forward(self, x):
        return self.linear(x)

    def fit(self, X, y, sample_weight=None):
        raise NotImplementedError("Subclasses must implement this method")

    def predict(self, X):
        raise NotImplementedError("Subclasses must implement this method")

    def summary(self):
        if self.results_ is None:
            raise ValueError("Model has not been fit yet.")
        return self.results_
