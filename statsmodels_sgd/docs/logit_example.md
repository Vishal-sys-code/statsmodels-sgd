# Logit Example

This example demonstrates how to use the Logit model from the Statsmodels-SGD package and compares it with statsmodels Logit.

```python
import numpy as np
import matplotlib.pyplot as plt
from statsmodels_sgd import Logit
import statsmodels.api as sm
from sklearn.metrics import roc_curve, auc

# Generate sample data
np.random.seed(42)
X = np.random.randn(1000, 5)
true_weights = np.array([1, -0.5, 0.25, -0.1, 0.2])
z = np.dot(X, true_weights) + np.random.randn(1000) * 0.1
y = (1 / (1 + np.exp(-z)) >= 0.5).astype(int)

# Generate sample weights
sample_weight = np.random.uniform(0.5, 1.5, size=1000)

# Fit our Logit model
our_model = Logit(n_features=X.shape[1]+1)
our_model.fit(X, y, sample_weight=sample_weight)
print("Our Logit Model Summary:")
print(our_model.summary())

# Fit statsmodels Logit
X_sm = sm.add_constant(X)
sm_model = sm.GLM(y, X_sm, family=sm.families.Binomial(), freq_weights=sample_weight).fit()
print("\nStatsmodels Logit Summary:")
print(sm_model.summary())

# Compare coefficients
our_coef = our_model.results_['coef']
sm_coef = sm_model.params

plt.figure(figsize=(10, 6))
plt.bar(range(len(our_coef)), our_coef, alpha=0.5, label='Our Logit')
plt.bar(range(len(sm_coef)), sm_coef, alpha=0.5, label='Statsmodels Logit')
plt.xlabel('Coefficient Index')
plt.ylabel('Coefficient Value')
plt.title('Comparison of Coefficients')
plt.legend()
plt.show()

# Compare predictions
our_pred = our_model.predict(X)
sm_pred = sm_model.predict(X_sm)

plt.figure(figsize=(10, 6))
plt.scatter(y, our_pred, alpha=0.5, label='Our Logit')
plt.scatter(y, sm_pred, alpha=0.5, label='Statsmodels Logit')
plt.xlabel('True Values')
plt.ylabel('Predicted Probabilities')
plt.title('True Values vs Predicted Probabilities')
plt.legend()
plt.show()

# Compare ROC curves
our_fpr, our_tpr, _ = roc_curve(y, our_pred, sample_weight=sample_weight)
sm_fpr, sm_tpr, _ = roc_curve(y, sm_pred, sample_weight=sample_weight)

our_auc = auc(our_fpr, our_tpr)
sm_auc = auc(sm_fpr, sm_tpr)

plt.figure(figsize=(10, 6))
plt.plot(our_fpr, our_tpr, label=f'Our Logit (AUC = {our_auc:.2f})')
plt.plot(sm_fpr, sm_tpr, label=f'Statsmodels Logit (AUC = {sm_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()

print(f"\nOur Logit AUC: {our_auc:.4f}")
print(f"Statsmodels Logit AUC: {sm_auc:.4f}")
```
