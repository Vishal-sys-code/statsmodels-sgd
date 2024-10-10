from .base_model import BaseModel
from .tools import (
    add_constant,
    calculate_standard_errors,
    calculate_t_p_values,
)
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd


class OLS(BaseModel):
    def fit(self, X, y, sample_weight=None, add_constant=True):
        if add_constant:
            X = add_constant(X)

        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

        if sample_weight is not None:
            sample_weight = torch.tensor(
                sample_weight, dtype=torch.float32
            ).reshape(-1, 1)

        optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)

        for _ in range(self.epochs):
            perm = torch.randperm(X.size(0))
            for i in range(0, X.size(0), self.batch_size):
                indices = perm[i : i + self.batch_size]
                X_batch, y_batch = X[indices], y[indices]

                if sample_weight is not None:
                    weight_batch = sample_weight[indices]
                else:
                    weight_batch = None

                optimizer.zero_grad()
                y_pred = self(X_batch)
                loss = self.weighted_mse_loss(y_pred, y_batch, weight_batch)
                loss.backward()
                torch.nn.utils.clip_grad_value_(
                    self.parameters(), self.clip_value
                )
                optimizer.step()

        params = (
            torch.cat([self.linear.bias, self.linear.weight.squeeze()])
            .detach()
            .numpy()
        )
        std_errors = calculate_standard_errors(
            X.numpy(),
            y.numpy(),
            params[1:],
            params[0],
            sample_weight=(
                sample_weight.numpy() if sample_weight is not None else None
            ),
        )
        t_values, p_values = calculate_t_p_values(params, std_errors)

        results = pd.DataFrame(
            {
                "coef": params,
                "std err": std_errors,
                "t": t_values,
                "P>|t|": p_values,
                "[0.025": params - 1.96 * std_errors,
                "0.975]": params + 1.96 * std_errors,
            },
            index=["const"] + [f"x{i}" for i in range(1, len(params))],
        )

        self.results_ = results
        return self

    def weighted_mse_loss(self, y_pred, y_true, weights=None):
        if weights is None:
            return nn.MSELoss()(y_pred, y_true)
        else:
            return torch.mean(weights * (y_pred - y_true) ** 2)

    def predict(self, X):
        if self.results_ is None:
            raise ValueError("Model has not been fit yet.")

        X = add_constant(X)
        X = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            return self(X).numpy()
