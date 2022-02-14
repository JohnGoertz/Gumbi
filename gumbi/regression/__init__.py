from .GP_pymc3 import GP
from .GP_gpflow import GP_gpflow
from .GP_gpflow import LVMOGP_GP
from .base import Regressor


__all__ = ['GP', 'GP_gpflow', 'LVMOGP_GP', 'Regressor']
