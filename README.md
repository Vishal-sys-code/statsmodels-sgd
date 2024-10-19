# statsmodels-sgd

Reimplementation of statsmodels using stochastic gradient descent

Use just like statsmodels, but with stochastic gradient descent.

## Installation

```bash
pip install git+https://github.com/PolicyEngine/statsmodels-sgd.git
```

## Example

```python
import statsmodels_sgd.api as sm_sgd

# Fit OLS model

model = sm_sgd.OLS(n_features=X.shape[1])
model.fit(X, y)

# Fit Logit model

model = sm_sgd.Logit(n_features=X.shape[1])
model.fit(X, y)
```
