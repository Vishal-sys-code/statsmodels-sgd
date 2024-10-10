# Example usage
if __name__ == "__main__":
    from statsmodels_sgd import OLS, Logit
    import statsmodels.api as sm
    import pandas as pd

    # OLS example with sample weights
    np.random.seed(42)
    X = np.random.randn(1000, 5)
    true_weights = np.array([1, 2, 3, 4, 5])
    y = np.dot(X, true_weights) + np.random.randn(1000) * 0.1
    sample_weight = np.random.uniform(0.5, 1)
