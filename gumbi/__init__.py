"""Gaussian Process Model Building Interface"""

from gumbi import data, style, utils

from .aggregation import *
from .arrays import *
from .plotting import *
from .regression import *
from .versions import __version__

# Aliases
parray = ParameterArray
uarray = UncertainArray
uparray = UncertainParameterArray
mvuparray = MVUncertainParameterArray
