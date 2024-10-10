# OLS Example

This example demonstrates how to use the OLS model from the Statsmodels-SGD package and compares it with statsmodels OLS.

```python
import numpy as np
import matplotlib.pyplot as plt
from statsmodels_sgd import OLS
import statsmodels.api as sm

# Generate sample data
np.random.seed(42)
X = np.random.randn(1000, 5)
true_weights = np.array([1, 2, 3, 4, 5])
y = np.dot(X, true_weights) + np.random.randn(1000) * 0.1

# Generate sample weights
sample_weight = np.random.uniform(0.5, 1.5, size=1000)

# Fit our OLS model
our_model = OLS(n_features=X.shape[1]+1)
our_model.fit(X, y, sample_weight=sample_weight)
print("Our OLS Model Summary:")
print(our_model.summary())

# Fit statsmodels OLS
X_sm = sm.add_constant(X)
sm_model = sm.WLS(y, X_sm, weights=sample_weight).fit()
print("\nStatsmodels OLS Summary:")
print(sm_model.summary())

# Compare coefficients
our_coef = our_model.results_['coef']
sm_coef = sm_model.params

plt.figure(figsize=(10, 6))
plt.bar(range(len(our_coef)), our_coef, alpha=0.5, label='Our OLS')
plt.bar(range(len(sm_coef)), sm_coef, alpha=0.5, label='Statsmodels OLS')
plt.xlabel('Coefficient Index')
plt.ylabel('Coefficient Value')
plt.title('Comparison of Coefficients')
plt.legend()
plt.show()

# Compare predictions
our_pred = our_model.predict(X)
sm_pred = sm_model.predict(X_sm)

plt.figure(figsize=(10, 6))
plt.scatter(y, our_pred, alpha=0.5, label='Our OLS')
plt.scatter(y, sm_pred, alpha=0.5, label='Statsmodels OLS')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs Predicted Values')
plt.legend()
plt.show()

# Compare R-squared
our_r2 = 1 - np.average((y - our_pred.squeeze())**2, weights=sample_weight) / np.average((y - np.average(y, weights=sample_weight))**2, weights=sample_weight)
sm_r2 = sm_model.rsquared

print(f"\nOur OLS R-squared: {our_r2:.4f}")
print(f"Statsmodels OLS R-squared: {sm_r2:.4f}")
```
