from .GP_pymc3 import GP
from .GP_gpflow import GP_gpflow
from .base import Regressor


__all__ = ['Regressor', 'GP', 'GP_gpflow']
