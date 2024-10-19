from .regression.linear_model import OLS
from .regression.discrete_model import Logit

# Make commonly used objects available at api level
__all__ = ["OLS", "Logit"]
