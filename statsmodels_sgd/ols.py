import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import statsmodels.api as sm


class OLS(nn.Module):
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

    def forward(self, x):
        return self.linear(x)

    def fit(self, X, y, sample_weight=None, add_constant=True):
        if add_constant:
            X = sm.add_constant(X)

        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

        if sample_weight is not None:
            sample_weight = torch.tensor(
                sample_weight, dtype=torch.float32
            ).reshape(-1, 1)

        optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)

        for _ in range(self.epochs):
            optimizer.zero_grad()
            y_pred = self(X)
            loss = self.weighted_mse_loss(y_pred, y, sample_weight)
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.parameters(), self.clip_value)
            optimizer.step()

    def weighted_mse_loss(self, y_pred, y_true, weights=None):
        if weights is None:
            return nn.MSELoss()(y_pred, y_true)
        else:
            return torch.mean(weights * (y_pred - y_true) ** 2)

    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            return self(X).numpy()
