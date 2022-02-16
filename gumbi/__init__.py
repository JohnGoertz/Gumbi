"""Gaussian Process Model Building Interface"""

from .versions import __version__

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


