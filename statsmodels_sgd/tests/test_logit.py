import numpy as np
import pytest
from statsmodels_sgd import Logit
import statsmodels.api as sm


@pytest.fixture
def sample_data():
    np.random.seed(42)
    X = np.random.randn(1000, 5)
    true_weights = np.array([1, -0.5, 0.25, -0.1, 0.2])
    z = np.dot(X, true_weights) + np.random.randn(1000) * 0.1
    y = (1 / (1 + np.exp(-z)) >= 0.5).astype(int)
    sample_weight = np.random.uniform(0.5, 1.5, size=1000)
    return X, y, sample_weight


def test_logit_fit_predict_with_weights(sample_data):
    X, y, sample_weight = sample_data
    model = Logit(n_features=X.shape[1] + 1)  # +1 for constant
    model.fit(X, y, sample_weight=sample_weight)

    # Check if model has results
    assert model.results_ is not None

    # Check predictions
    y_pred = model.predict(X)
    y_pred_binary = (y_pred >= 0.5).astype(int).flatten()
    accuracy = np.mean(y_pred_binary == y)
    assert accuracy > 0.7  # Adjust this threshold as needed


def test_logit_vs_statsmodels_with_weights(sample_data):
    X, y, sample_weight = sample_data

    # Fit our model
    our_model = Logit(n_features=X.shape[1] + 1)  # +1 for constant
    our_model.fit(X, y, sample_weight=sample_weight)

    # Fit statsmodels Logit
    X_sm = sm.add_constant(X)
    sm_model = sm.GLM(
        y, X_sm, family=sm.families.Binomial(), freq_weights=sample_weight
    ).fit()

    # Compare coefficients
    np.testing.assert_allclose(
        our_model.results_["coef"], sm_model.params, rtol=0.1, atol=0.1
    )

    # Compare predictions
    our_pred = our_model.predict(X)
    sm_pred = sm_model.predict(X_sm)
    np.testing.assert_allclose(our_pred.squeeze(), sm_pred, rtol=0.1)

    # Print coefficients and summary for debugging
    print("Our model coefficients:", our_model.results_["coef"])
    print("Statsmodels coefficients:", sm_model.params)
    print("\nOur model summary:")
    print(our_model.summary())
    print("\nStatsmodels summary:")
    print(sm_model.summary())
