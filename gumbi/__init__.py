"""Gaussian Process Model Building Interface"""

__version__ = '0.1.0'

from .regression import *
from .aggregation import *
from .arrays import *
from .plotting import *
from gumbi import style, utils, data

# Aliases
parray = ParameterArray
uarray = UncertainArray
uparray = UncertainParameterArray
mvuparray = MVUncertainParameterArray


