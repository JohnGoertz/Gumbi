from __future__ import annotations  # Necessary for self-type annotations until Python >3.10

import copy
import pickle
import warnings
import numpy as np
import pandas as pd

from scipy.stats import norm, lognorm, chi2, ncx2, rv_continuous, multivariate_normal
from scipy.stats._multivariate import multivariate_normal_frozen
from scipy.special import logit, expit

from uncertainties import unumpy as unp

from .utils import skip
from .aggregation import Standardizer

__all__ = ['LayeredArray', 'ParameterArray', 'UncertainArray', 'UncertainParameterArray', 'MVUncertainParameterArray']


class LogitNormal(rv_continuous):
    r"""A logit-normal continuous random variable.

    The probability density function for LogitNormal is:

    .. math::
        f \left( x, \sigma \right) = \frac{1}{\sigma \sqrt{2 \pi}} e^{- \frac{ \left( \text{logit} (x) - \mu \right)^2}{2 \sigma^2}} \frac{1}{x \left( 1-x \right)}

    for :math:`0<x<1`, :math:`\sigma>0`

    A logit-normal random variable `Y` can be parameterized in terms of the mean, :math:`\mu`, and standard deviation,
    :math:`\sigma`, of the unique normally distributed random variable `X` such that `expit(X) = Y`. This parametrization
    corresponds to setting `scale = σ` and `loc = expit(μ)`.
    """
    def __init__(self, loc=0.5, scale=1):
        super().__init__(self)
        self.scale = scale
        self.loc = logit(loc)

    def _pdf(self, x):
        return norm(loc=self.loc, scale=self.scale).pdf(logit(x)) / (x * (1 - x))

    def _cdf(self, x):
        return norm(loc=self.loc, scale=self.scale).cdf(logit(x))

    def ppf(self, q):
        return expit(norm(loc=self.loc, scale=self.scale).ppf(q))

    def rvs(self, size=None, random_state=None):
        return expit(norm(loc=self.loc, scale=self.scale).rvs(size=size, random_state=random_state))


class MultivariateNormalish(multivariate_normal_frozen):
    r"""A multivariate Normal distribution built from and callable on ParameterArrays.

    In particular, this class takes care of transforming variables to/from "natural" space as necessary.

    Parameters
    ----------
    mean : ParameterArray
        A ParameterArray containing the distribution mean, must be a single point
    cov : float or np.array, default 1
        Covariance matrix of the distribution as a standard array or scalar (default one)
    **kwargs
        Additional keyword arguments passed to `scipy.stats.multivariate_normal`
    """

    def __init__(self, mean: ParameterArray, cov: int | float | np.ndarray, **kwargs):
        assert isinstance(mean, ParameterArray), 'Mean must be a ParameterArray'
        if mean.ndim != 0:
            raise NotImplementedError('Multidimensional multivariate distributions are not yet supported.')

        self._names = mean.names
        self._stdzr = mean.stdzr
        self._log_vars = mean.stdzr.log_vars
        self._logit_vars = mean.stdzr.logit_vars
        self._islog = [1 if var in self._log_vars else 0 for var in self._names]
        self._islogit = [1 if var in self._logit_vars else 0 for var in self._names]

        super(MultivariateNormalish, self).__init__(mean=mean.z.values(), cov=cov, **kwargs)

    def pdf(self, x: ParameterArray) -> float:
        r"""Probability density function.

        Parameters
        ----------
        x : ParameterArray
            Quantiles, with the last axis of x denoting the components.

        Returns
        -------
        array-like
        """
        return super(MultivariateNormalish, self).pdf(x)

    def logpdf(self, x: ParameterArray) -> float:
        r"""Log of the probability density function.

        Parameters
        ----------
        x : ParameterArray
            Quantiles, with the last axis of x denoting the components.

        Returns
        -------
        array-like
        """
        return super(MultivariateNormalish, self).logpdf(x.z.dstack())

    def cdf(self, x: ParameterArray) -> float:
        r"""Cumulative distribution function.

        Parameters
        ----------
        x : ParameterArray
            Quantiles, with the last axis of x denoting the components.

        Returns
        -------
        array-like
        """
        return super(MultivariateNormalish, self).cdf(x.z.dstack())

    def logcdf(self, x: ParameterArray) -> float:
        r"""Log of the cumulative distribution function.

        Parameters
        ----------
        x : ParameterArray
            Quantiles, with the last axis of x denoting the components.

        Returns
        -------
        array-like
        """
        return super(MultivariateNormalish, self).logcdf(x)

    def rvs(self, size=1, random_state=None) -> ParameterArray:
        r"""Draw random samples from the distribution.

        Parameters
        ----------
        size: int or array-like of ints
        random_state: int

        Returns
        -------
        ParameterArray
        """
        # multivariate_normal throws a RuntimeWarning: covariance is not positive semidefinite if rvs() is called more
        # than once without calling dist().rvs() in between. Covariance isn't being altered as a side-effect of this
        # call, so this issue really makes no sense.....
        # with warnings.catch_warnings():
        #     warnings.filterwarnings("error")
        #     try:
        #         samples = super(MultivariateNormalish, self).rvs(size=size, random_state=random_state)
        #     except:
        #         multivariate_normal(mean=self.mean, cov=self.cov).rvs()
        #         samples = super(MultivariateNormalish, self).rvs(size=size, random_state=random_state)
        samples = super(MultivariateNormalish, self).rvs(size=size, random_state=random_state)
        return ParameterArray(**{p: samples[..., i] for i, p in enumerate(self._names)}, stdzd=True, stdzr=self._stdzr)


class LayeredArray(np.ndarray):
    """An array with one or more named values at every index.

    Parameters
    ----------
    name : str
    array : array-like
    """
    def __new__(cls, stdzr=None, **arrays):
        if arrays == {}:
            raise ValueError('Must supply at least one array')
        arrays = {name: np.asarray(array) for name, array in arrays.items() if array is not None}

        narray_dtype = np.dtype([(name, array.dtype) for name, array in arrays.items()])

        narray_prototype = np.empty(list(arrays.values())[0].shape, dtype=narray_dtype)
        for name, array in arrays.items():
            narray_prototype[name] = array

        larray = narray_prototype.view(cls)
        larray.names = list(narray_dtype.fields.keys())
        larray.stdzr = stdzr

        return larray

    def __array_finalize__(self, larray):
        if larray is None:
            return
        self.names = getattr(larray, 'names', None)
        self.stdzr = getattr(larray, 'stdzr', None)

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        """https://numpy.org/doc/stable/user/basics.subclassing.html#array-ufunc-for-ufuncs"""
        args = []
        if len(set([larray.names[0] for larray in inputs if isinstance(larray, LayeredArray)]))>1:
            warnings.warnings.warn('Operating on arrays with different layer names, results may be unexpected.')
        for input_ in inputs:
            if isinstance(input_, LayeredArray):
                if len(input_.names) > 1:
                    raise ValueError('Cannot operate on array with multiple layer names')
                args.append(input_.astype(float).view(np.ndarray))
            else:
                args.append(input_)

        outputs = kwargs.get('out')
        if outputs:
            out_args = []
            for output in outputs:
                if isinstance(output, LayeredArray):
                    out_args.append(output.astype(float).view(np.ndarray))
                else:
                    out_args.append(output)
            kwargs['out'] = tuple(out_args)
        else:
            outputs = (None,) * ufunc.nout

        results = super().__array_ufunc__(ufunc, method, *args, **kwargs)

        if results is NotImplemented:
            return NotImplemented

        if ufunc.nout == 1:
            results = (results,)

        results = tuple(LayeredArray(**{self.names[0]: result})
                        if output is None else output
                        for result, output in zip(results,outputs))

        return results[0] if len(results) == 1 else results

    def __getitem__(self, item):
        default = super().__getitem__(item)
        if isinstance(item, str):
            arrays = {item: default}
        elif isinstance(item, int) or (isinstance(item, tuple) and all(isinstance(val, int) for val in item)):
            arrays = {name: value for name, value in zip(default.dtype.names, default)}
        elif isinstance(item, slice):
            arrays = {layer.names[0]: layer.values() for layer in default.as_list()}
        else:
            return default
        return LayeredArray(**arrays)

    def __repr__(self):
        return f'{tuple(self.names)}: {np.asarray(self)}'

    def get(self, name, default=None):
        """Return value given by `name` if it exists, otherwise return `default`."""
        if name in self.names:
            return self[name]
        elif default is None:
            return None
        else:
            return LayeredArray(**{name: default})

    def drop(self, name, missing_ok=True):
        if name in self.names:
            return LayeredArray(**{p: arr for p, arr in self.as_dict().items() if p != name})
        elif missing_ok:
            return self
        else:
            raise KeyError(f'Name {name} not found in array.')

    def values(self):
        """Values at each index stacked into regular ndarray."""
        stacked = np.stack([self[name].astype(float) for name in self.names])
        if len(self.names) > 1:
            return stacked
        else:
            return stacked[0]

    def dstack(self):
        """Values at each index as ndarrays stacked in sequence depth wise (along third axis)."""
        return np.dstack([la.values() for la in self.as_list()])

    def as_list(self, order=None):
        order = self.names if order is None else order
        assert all(name in order for name in self.names)
        return [self[name] for name in order]

    def as_dict(self):
        """Values corresponding to each named level as a dictionary."""
        return {name: self[name].values() for name in self.names}

    def add_layers(self, **arrays):
        """Add additional layers at each index."""
        arrays_ = arrays.as_dict() if isinstance(arrays, LayeredArray) else arrays
        return LayeredArray(**(self.as_dict() | arrays_))


class ParameterArray(LayeredArray):
    """Array of parameter values, allowing simple transformation.

    :class:`ParameterArray` stores not only the value of the variable itself but also a :class:`Standardizer` instance.
    This makes it simple to switch between the natural scale of the parameter and its transformed and standardized
    values through the :attr:`t` and :attr:`z` properties, respectively.

    This class can also be accessed through the alias :class:`parray`.

    Parameters
    ----------
    **arrays
        arrays to store with their names as keywords
    stdzr : Standardizer
        An instance  of :class:`Standardizer`
    stdzd : bool, default False
        Whether the supplied values are on standardized scale instead of the natural scale

    Examples
    --------

    A parray can created with a single parameter. In this case, `r` is treated as a `LogNormal` variable by the stdzr.

    >>> from gumbi import ParameterArray as parray
    >>> stdzr = Standardizer(d={'μ': -0.307, 'σ': 0.158}, log_vars=['d'])
    >>> rpa = parray(d=np.arange(5,10)/10, stdzr=stdzr)
    >>> rpa
    ('d',): [(0.5,) (0.6,) (0.7,) (0.8,) (0.9,)]
    >>> rpa.t
    ('r_t',): [(-0.69314718,) (-0.51082562,) (-0.35667494,) (-0.22314355,) (-0.10536052,)]
    >>> rpa.z
    ('r_z',): [(-2.4439695 ,) (-1.29003559,) (-0.31439838,) ( 0.53073702,) ( 1.27619927,)]

    If the parameter is completely absent from the stdzr, its natural, :attr:`t`, and :attr:`z` values are identical.

    >>> pa = parray(param=np.arange(5), stdzr=stdzr)
    >>> pa
    ('param',): [(0,) (1,) (2,) (3,) (4,)]
    >>> pa.t
    ('param_t',): [(0,) (1,) (2,) (3,) (4,)]
    >>> pa.z
    ('param_z',): [(0.,) (1.,) (2.,) (3.,) (4.,)]

    You can even do monstrous compositions like

    >>> np.min(np.sqrt(np.mean(np.square(rpa-rpa[0]-0.05)))).t
    ('r_t',): (-1.5791256,)

    If you `have` work with an ordinary numpy array, use :meth:`values`.

    >>> np.argmax(rpa.values())
    4

    Attributes
    ----------
    names : list of str
        Names of all stored parameters
    stdzr : Standardizer
        An instance  of :class:`Standardizer`
    """

    def __new__(cls, stdzr: Standardizer, stdzd=False, **arrays):
        if arrays == {}:
            raise ValueError('Must supply at least one array')

        if stdzd:
            arrays = {name: stdzr.unstdz(name, np.array(array))
                      for name, array in arrays.items()}

        parray = LayeredArray.__new__(cls, **arrays)
        parray.stdzr = stdzr

        return parray

    def __array_finalize__(self, parray):
        if parray is None:
            return
        self.stdzr = getattr(parray, 'stdzr', None)
        self.names = getattr(parray, 'names', None)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        result = super().__array_ufunc__(ufunc, method, *inputs, **kwargs)
        if result is NotImplemented:
            return NotImplemented
        return ParameterArray(**result.as_dict(), stdzr=self.stdzr, stdzd=False)

    def __getitem__(self, item):
        default = super(LayeredArray, self).__getitem__(item)
        if isinstance(item, str):
            arrays = {item: default}
        elif isinstance(item, int) or (isinstance(item, tuple) and all(isinstance(val, int) for val in item)):
            arrays = {name: value for name, value in zip(default.dtype.names, default)}
        elif isinstance(item, slice):
            arrays = {layer.names[0]: layer.values() for layer in default.as_list()}
        else:
            return default
        return ParameterArray(**arrays, stdzr=self.stdzr, stdzd=False)


    def get(self, name, default=None):
        """Return value given by `name` if it exists, otherwise return `default`"""
        if name in self.names:
            return self[name]
        elif default is None:
            return None
        else:
            return self.parray(**{name: default})

    def drop(self, name, missing_ok=True):
        if name in self.names:
            return self.parray(**{p: arr for p, arr in self.as_dict().items() if p != name})
        elif missing_ok:
            return self
        else:
            raise KeyError(f'Name {name} not found in array.')

    @property
    def z(self) -> LayeredArray:
        """Standardized values"""
        zdct = {name+'_z': self.stdzr.stdz(name, self[name].values()) for name in self.names}
        return LayeredArray(**zdct, stdzr=self.stdzr)

    @property
    def t(self) -> LayeredArray:
        """Transformed values"""
        tdct = {name+'_t': self.stdzr.transform(name, self[name].values()) for name in self.names}
        return LayeredArray(**tdct, stdzr=self.stdzr)

    def add_layers(self, stdzd=False, **arrays):
        """Add additional layers at each index"""
        narrays = super().add_layers(**arrays)
        if stdzd:
            for name in narrays.names:
                narrays[name] = self.stdzr.unstdz(name, narrays[name])
        return self.parray(**narrays.as_dict(), stdzd=False)

    def fill_with(self, **params):
        """Add a single value for a new parameter at all locations."""
        assert all([isinstance(value, (float, int)) for value in params.values()])
        assert all([isinstance(key, str) for key in params.keys()])
        return self.add_layers(**{param: np.full(self.shape, value) for param, value in params.items()})

    def parray(self, *args, **kwargs):
        """Create a new ParameterArray using this instance's standardizer"""
        return ParameterArray(*args, **kwargs, stdzr=self.stdzr)

    @classmethod
    def stack(cls, parray_list, axis=0, **kwargs):
        all_names = [pa.names for pa in parray_list]
        if not all(names == all_names[0] for names in all_names):
            raise ValueError('Arrays do not have the same names!')
        new = np.stack(parray_list, axis=axis, **kwargs)
        stdzr = parray_list[0].stdzr
        return cls(**{dim: new[dim] for dim in new.dtype.names}, stdzr=stdzr)

    @classmethod
    def vstack(cls, parray_list, **kwargs):
        all_names = [pa.names for pa in parray_list]
        if not all(names == all_names[0] for names in all_names):
            raise ValueError('Arrays do not have the same names!')
        new = np.vstack(parray_list, **kwargs)
        stdzr = parray_list[0].stdzr
        return cls(**{dim: new[dim] for dim in new.dtype.names}, stdzr=stdzr)

    @classmethod
    def hstack(cls, parray_list, **kwargs):
        all_names = [pa.names for pa in parray_list]
        if not all(names == all_names[0] for names in all_names):
            raise ValueError('Arrays do not have the same names!')
        new = np.hstack(parray_list, **kwargs)
        stdzr = parray_list[0].stdzr
        return cls(**{dim: new[dim] for dim in new.dtype.names}, stdzr=stdzr)


class UncertainArray(np.ndarray):
    """Structured array containing mean and variance of a normal distribution at each point.

    The main purpose of this object is to correctly `propagate uncertainty`_ under transformations. Arithmetic
    operations between distributions or between distributions and scalars are handled appropriately via the
    `uncertainties`_ package.

    Additionally, a `scipy Normal distribution`_ object can be created at each point through the :attr:`dist` property,
    allowing access to that objects such as :meth:`rvs`, :meth:`ppf`, :meth:`pdf`, etc.

    This class can also be accessed through the alias :class:`uarray`.

    .. _`propagate uncertainty`: https://en.wikipedia.org/wiki/Propagation_of_uncertainty
    .. _`uncertainties`: https://pythonhosted.org/uncertainties/
    .. _`scipy Normal distribution`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html

    Notes
    -----
    The `name` argument is intended to be the general name of the value held, not unique to this instance. Combining two
    :class:`UncertainArray` objects with the same name results in a new object with that name; combining two objects
    with different names results in a new name that reflects this combination (so ``'A'+'B'`` becomes ``'(A+B)'``).

    Parameters
    ----------
    name : str
        Name of variable.
    μ : array
        Mean at each point
    σ2 : array
        Variance at each point
    **kwargs
        Names and values of additional arrays to store

    Examples
    --------
    >>> ua1 = UncertainArray('A', μ=1, σ2=0.1)
    >>> ua2 = uarray('A', μ=2, σ2=0.2)  #equivalent
    >>> ua2
    A['μ', 'σ2']: (2, 0.2)

    Addition of a scalar

    >>> ua1+1
    A['μ', 'σ2']: (2, 0.1)

    Addition and subtraction of two UncertainArrays:

    >>> ua2+ua1
    A['μ', 'σ2']: (3., 0.3)
    >>> ua2-ua1
    A['μ', 'σ2']: (1., 0.3)

    Note, however, that correlations are not properly accounted for (yet). Subtracting one UncertainArray from itself
    should give exactly zero with no uncertainty, but it doesn't:

    >>> ua2+ua2
    A['μ', 'σ2']: (4., 0.4)
    >>> ua2-ua2
    A['μ', 'σ2']: (0., 0.4)

    Mean of two `uarray` objects:

    >>> uarray.stack([ua1, ua2]).mean(axis=0)
    A['μ', 'σ2']: (1.5, 0.075)

    Mean within a single `uarray` object:

    >>> ua3 = uarray('B', np.arange(1,5)/10, np.arange(1,5)/100)
    >>> ua3
    B['μ', 'σ2']: [(0.1, 0.01) (0.2, 0.02) (0.3, 0.03) (0.4, 0.04)]
    >>> ua3.μ
    array([0.1, 0.2, 0.3, 0.4])
    >>> ua3.mean()
    B['μ', 'σ2']: (0.25, 0.00625)

    Adding two `uarrays` with differnt name creates an object with a new name

    >>> ua1+ua3.mean()
    (A+B)['μ', 'σ2']: (1.25, 0.10625)

    Accessing :attr:`dist` methods

    >>> ua3.dist.ppf(0.95)
    array([0.26448536, 0.43261743, 0.58489701, 0.72897073])
    >>> ua3.dist.rvs([3,*ua3.shape])
    array([[0.05361942, 0.14164882, 0.14924506, 0.03808633],
           [0.05804824, 0.09946732, 0.08727794, 0.28091272],
           [0.06291355, 0.47451576, 0.20756356, 0.2108717 ]])  # random

    Attributes
    ----------
    name : str
        Name of variable.
    fields : list of str
        Names of each level held in the array
    """

    def __new__(cls, name: str, μ: np.ndarray, σ2: np.ndarray, stdzr=None, **kwargs):
        μ_ = np.asarray(μ)
        σ2_ = np.asarray(σ2)
        assert(μ_.shape == σ2_.shape)
        base_dtypes = [('μ', μ_.dtype), ('σ2', σ2_.dtype)]
        extra_dtypes = [(dim, np.asarray(arr).dtype) for dim, arr in kwargs.items() if arr is not None]
        uarray_dtype = np.dtype(base_dtypes+extra_dtypes)

        uarray_prototype = np.empty(μ_.shape, dtype=uarray_dtype)
        uarray_prototype['μ'] = μ_
        uarray_prototype['σ2'] = σ2_
        for dim, arr in kwargs.items():
            if arr is not None:
                uarray_prototype[dim] = np.asarray(arr)

        uarray = uarray_prototype.view(cls)
        uarray.name = name
        uarray.stdzr = stdzr
        uarray.fields = list(uarray_dtype.fields.keys())

        return uarray

    def __array_finalize__(self, uarray):
        if uarray is None:
            return
        self.name = getattr(uarray, 'name', None)
        self.stdzr = getattr(uarray, 'stdzr', None)
        self.fields = getattr(uarray, 'fields', None)

    @property
    def μ(self) -> np.ndarray:
        """Nominal value (mean)"""
        return self['μ']

    @μ.setter
    def μ(self, val):
        self['μ'] = val

    @property
    def σ2(self) -> np.ndarray:
        """Variance"""
        return self['σ2']

    @σ2.setter
    def σ2(self, val):
        self['σ2'] = val

    @property
    def σ(self) -> np.ndarray:
        """Standard deviation"""
        return np.sqrt(self.σ2)

    @σ.setter
    def σ(self, val):
        self['σ2'] = val**2

    @property
    def _as_uncarray(self):
        return unp.uarray(self.μ, self.σ)

    @classmethod
    def _from_uncarray(cls, name, uncarray, **extra):
        return cls(name=name, μ=unp.nominal_values(uncarray), σ2=unp.std_devs(uncarray)**2, **extra)

    @property
    def dist(self) -> rv_continuous:
        """Array of :func:`scipy.stats.norm` objects"""
        return norm(loc=self.μ, scale=self.σ)

    @staticmethod
    def stack(uarray_list, axis=0) -> UncertainArray:
        new = np.stack(uarray_list, axis=axis)
        names = [ua.name for ua in uarray_list]
        if all(name == names[0] for name in names):
            name = uarray_list[0].name
        else:
            raise ValueError('Arrays do not have the same name!')
            # name = '('+', '.join(names)+')'
        return UncertainArray(name, **{dim: new[dim] for dim in new.dtype.names})

    def nlpd(self, target) -> float:
        """Negative log posterior density"""
        return -np.log(self.dist.pdf(target))

    def EI(self, target, best_yet, k=1) -> float:
        """Expected improvement

        Taken from https://github.com/akuhren/target_vector_estimation

        Parameters
        ----------
        target : float
        best_yet : float
        k : int

        Returns
        -------
        EI : float
        """

        nc = ((target - self.μ) ** 2) / self.σ2

        h1_nx = ncx2.cdf((best_yet / self.σ2), k, nc)
        h2_nx = ncx2.cdf((best_yet / self.σ2), (k+2), nc)
        h3_nx = ncx2.cdf((best_yet / self.σ2), (k+4), nc)

        t1 = best_yet * h1_nx
        t2 = self.σ2 * (k * h2_nx + nc * h3_nx)

        return t1 - t2

    def KLD(self, other):
        """Kullback–Leibler Divergence

        Parameters
        ----------
        other : UncertainArray

        Returns
        -------
        divergence : float
        """
        assert isinstance(other, UncertainArray)
        return np.log(other.σ / self.σ) + (self.σ2 + (self.μ - other.μ) ** 2) / (2 * other.σ2) - 1 / 2

    def BD(self, other):
        """Bhattacharyya Distance

        Parameters
        ----------
        other : UncertainArray

        Returns
        -------
        distance : float
        """
        assert isinstance(other, UncertainArray)
        return 1 / 4 * np.log(1 / 4 * (self.σ2 / other.σ2 + other.σ2 / self.σ2 + 2)) + 1 / 4 * ((self.μ - other.μ) ** 2 / (self.σ2 + other.σ2))


    def BC(self, other):
        """Bhattacharyya Coefficient

        Parameters
        ----------
        other : UncertainArray

        Returns
        -------
        coefficient : float
        """
        return np.exp(-self.BD(other))


    def HD(self, other):
        """Hellinger Distance

        Parameters
        ----------
        other : UncertainArray

        Returns
        -------
        distance : float
        """
        return np.sqrt(1 - self.BC(other))

    def __repr__(self):
        return f'{self.name}{self.fields}: {np.asarray(self)}'

    def __getitem__(self, item):
        default = super().__getitem__(item)
        if isinstance(item, int) or (isinstance(item, tuple) and all(isinstance(val, int) for val in item)):
            arrays = {name: value for name, value in zip(default.dtype.names, default)}
        elif isinstance(item, slice):
            # arrays = {layer.names[0]: layer.values() for layer in default.as_list()}
            return default
        else:
            return default.view(np.ndarray)
        return UncertainArray(self.name, **arrays)

    def sum(self, axis=None, dtype=None, out=None, keepdims=False, **kwargs) -> UncertainArray:
        """Summation with uncertainty propagation"""
        kwargs |= dict(axis=axis, dtype=dtype, out=out, keepdims=keepdims)
        new = self._as_uncarray.sum(**kwargs)
        extra = {dim: np.sum(self[dim]) for dim in self.fields if dim not in ['μ', 'σ2']}
        return self._from_uncarray(self.name, new, **extra)

    def mean(self, axis=None, dtype=None, out=None, keepdims=False, **kwargs) -> UncertainArray:
        """Mean with uncertainty propagation"""
        kwargs |= dict(axis=axis, dtype=dtype, out=out, keepdims=keepdims)
        new = self._as_uncarray.mean(**kwargs)
        extra = {dim: np.mean(self[dim]) for dim in self.fields if dim not in ['μ', 'σ2']}
        return self._from_uncarray(self.name, new, **extra)

    def __add__(self, other):
        new = self._as_uncarray
        if isinstance(other, UncertainArray):
            new += other._as_uncarray
            name = self.name if self.name == other.name else f'({self.name}+{other.name})'
        else:
            new += other
            name = self.name
        extra = {dim: np.mean(self[dim]) for dim in self.fields if dim not in ['μ', 'σ2']}
        return self._from_uncarray(name, new, **extra)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        new = self._as_uncarray
        if isinstance(other, UncertainArray):
            new -= other._as_uncarray
            name = self.name if self.name == other.name else f'({self.name}+{other.name})'
        else:
            new -= other
            name = self.name
        extra = {dim: np.mean(self[dim]) for dim in self.fields if dim not in ['μ', 'σ2']}
        return self._from_uncarray(name, new, **extra)

    def __rsub__(self, other):
        if isinstance(other, UncertainArray):
            new = other._as_uncarray
            name = self.name if self.name == other.name else f'({self.name}+{other.name})'
        else:
            new = copy.copy(other)
            name = self.name
        new -= self._as_uncarray
        extra = {dim: np.mean(self[dim]) for dim in self.fields if dim not in ['μ', 'σ2']}
        return self._from_uncarray(name, new, **extra)


class UncertainParameterArray(UncertainArray):
    r"""Structured array of parameter means and variances, allowing transformation with uncertainty handling.

    The primary role of this class is to compactly store the outputs of our regression models
    (e.g., :class:`gumbi.GP`). We typically use these models
    to produce parameter predictions or estimates, but under some transformation. For example, reaction `rate` must
    clearly be strictly positive, so we fit a GP to the `log` of rate in order to more appropriately conform to the
    assumption of normality. For prediction and visualization, however, we often need to switch back and forth between
    natural space (:math:`rate`), transformed space (:math:`\text{ln}\; rate`), and standardized space
    (:math:`\left( \text{ln}\; rate  - \mu_{\text{ln}\; rate} \right)/\sigma_{\text{ln}\; rate}`), meanwhile calculating
    summary statistics such as means and percentiles. This class is intended to facilitate switching between those
    different contexts.

    :class:`UncertainParameterArray`, also accessible through the alias :class:`uparray`, combines the functionality of
    :class:`ParameterArray` and :class:`UncertainArray`. A `uparray` stores the mean and variance of the
    variable itself as well as a :class:`Standardizer`
    instance. This  makes it simple to switch between the natural scale of the parameter and its transformed and
    standardized values through the :attr:`t` and :attr:`z` properties, respectively, with the accompanying variance
    transformed and scaled appropriately. This uncertainty is propagated under transformation, as with
    :class:`UncertainArray`, and a scipy distribution object can be created at each point through the :attr:`dist`
    property, allowing access to that objects such as :meth:`rvs`, :meth:`ppf`, :meth:`pdf`, etc.

    Notes
    -----
    The `name` argument is intended to be the general name of the value held, not unique to this instance. Combining two
    :class:`UncertainParameterArray` objects with the same name results in a new object with that name; combining two
    objects with different names results in a new name that reflects this combination (so ``'A'+'B'`` becomes
    ``'(A+B)'``).

    The behavior of this object depends on the transformation associated with it, as indicated by its `name` in its
    stored :class:`Standardizer` instance. If this transformation is :func:`np.log`, the parameter is treated as a
    `LogNormal` variable; otherwise it's treated as a `Normal` variable. This affects which distribution is returned by
    :attr:`dist` (`lognorm`_ vs `norm`_) and also the interpretation of :attr:`μ` and :attr:`σ2`.

     * For a `Normal` random
       variable, these are simply parameter's mean and variance in unstandardized space, :attr:`t.μ` and :attr:`t.σ2`
       are identical to :attr:`μ` and :attr:`σ2`, and :attr:`z.μ` and :attr:`z.σ2` are the parameter's mean and variance
       in standardized space.
     * For a `LogNormal` random variable ``Y``, however, :attr:`t.μ` and :attr:`t.σ2` are the mean and variance of a
       `Normal` variable ``X`` such that ``exp(X)=Y`` (:attr:`z.μ` and :attr:`z.σ2` are this mean and variance in
       standardized space). In this case, :attr:`μ` and :attr:`σ2` are the scale and shape descriptors of ``Y``, so
       ``self.μ = np.exp(self.t.μ)`` and ``self.σ2 = self.t.σ2``. Thus, :attr:`μ` and :attr:`σ2` are not strictly the
       mean and variance of the random variable in natural space, these can be obtained from the :attr:`dist`.

       * This behavior is most important, and potentially most confusing, when calculating the :meth:`mean`. Averaging
         is performed in `transformed` space, where the random variable exhibits a `Normal` distribution and the mean
         also exhibits a `Normal` distribution, allowing error propagation to be applied analytically. The :attr:`μ` and
         :attr:`σ2` returned are the descriptors of the `LogNormal` distribution that represents the reverse
         transformation of this new `Normal` distribution. Therefore, the result is more akin to marginalizing out the
         given dimensions in the underlying model than a true natural-space average.

    .. _norm: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
    .. _lognorm: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html

    See Also
    --------
    `norm <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html>`_: \
    scipy `Normal` random variable

    `lognorm <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html>`_: \
    scipy `LogNormal` random variable

    :class:`ParameterArray`

    :class:`UncertainArray`

    :class:`Standardizer`

    Parameters
    ----------
    name : str
        Name of variable.
    μ : array
        Mean at each point
    σ2 : array
        Variance at each point
    stdzr : Standardizer
        An instance  of :class:`Standardizer`, converted internally to :class:`Standardizer`
    stdzd : bool, default False
        Whether the supplied values are on standardized scale instead of the natural scale

    Examples
    --------
    Create a `LogNormal` random variable, as indicated by its :class:`Standardizer`

    >>> from gumbi import uparray, Standardizer
    >>> import numpy as np
    >>> stdzr = Standardizer(m = {'μ': -5.30, 'σ': 0.582}, log_vars=['c'])
    >>> upa = uparray('c', np.arange(1,5)/10, np.arange(1,5)/100, stdzr)
    >>> upa
    m['μ', 'σ2']: [(0.1, 0.01) (0.2, 0.02) (0.3, 0.03) (0.4, 0.04)]
    >>> stdzr.transforms['c']
    [<ufunc 'log'>, <ufunc 'exp'>]

    Mean and variance of the parameter in standardized space:

    >>> upa.z
    m_z['μ', 'σ2']: [(5.15019743, 0.02952256) (6.34117197, 0.05904512)
                     (7.03784742, 0.08856768) (7.53214651, 0.11809024)]

    Verify round-trip transformation:

    >>> upa.stdzr.unstdz(upa.name, upa.z.μ, upa.z.σ2)
    (array([0.1, 0.2, 0.3, 0.4]), array([0.01, 0.02, 0.03, 0.04]))

    Create a `uparray` from already-standardized values and verify round-trip transformation:

    >>> uparray('c', np.arange(-2,3), np.arange(1,6)/10, stdzr, stdzd=True).z
    m_z['μ', 'σ2']: [(-2., 0.1) (-1., 0.2) ( 0., 0.3) ( 1., 0.4) ( 2., 0.5)]

    For `LogNormal` parameters, uparray follows the `scipy.stats` convention  of parameterizing a lognormal random
    variable in terms of it's natural-space mean and its log-space standard deviation. Thus, a LogNormal uparray defined
    as `m['μ', 'σ2']: (0.1, 0.01)` represents `exp(Normal(log(0.1), 0.01))`.

    Note that the mean is not simply the mean of each component, it is the parameters of the `LogNormal` distribution
    that corresponds to the mean of the underlying `Normal` distributions in `log` (transformed) space.

    >>> upa.μ.mean()
    0.25
    >>> upa.σ2.mean()
    0.025
    >>> upa.mean()
    m['μ', 'σ2']: (0.22133638, 0.00625)

    You can verify the mean and variance returned by averaging over the random variable explicitly.

    >>> upa.mean().dist.mean()
    2.2202914201059437e-01
    >>> np.exp(upa.t.mean().dist.rvs(10000, random_state=2021).mean())
    2.2133371283050837e-01
    >>> upa.mean().dist.var()
    3.0907071428047016e-04
    >>> np.log(upa.mean().dist.rvs(10000, random_state=2021)).var()
    6.304628046829242e-03

    Calculate percentiles

    >>> upa.dist.ppf(0.025)
    array([0.08220152, 0.1515835 , 0.21364308, 0.27028359])
    >>> upa.dist.ppf(0.975)
    array([0.12165225, 0.26388097, 0.42126336, 0.59197082])

    Draw samples

    >>> upa.dist.rvs([3, *upa.shape], random_state=2021)
    array([[0.11605116, 0.22006429, 0.27902589, 0.34041327],
           [0.10571616, 0.1810085 , 0.36491077, 0.45507622],
           [0.10106982, 0.21230397, 0.3065239 , 0.33827997]])

    You can compose the variable with numpy functions, though you may get a warning if the operation is poorly defined
    for the distribution (which is most transforms on `LogNormal` distributions). Transformations are applied in
    transformed space.

    >>> (upa+1+np.tile(upa, (3,1))[2,3]).mean().t.dist.ppf(0.5)
    UserWarning: Transform is poorly defined for <ufunc 'log'>; results may be unexpected.
    -1.8423623672812148


    Attributes
    ----------
    name : str
        Name of variable.
    μ : array
        Mean at each point
    σ2 : array
        Variance at each point
    fields : list of str
        Names of each level held in the array
    stdzr : Standardizer
        An instance  of :class:`Standardizer` created from the supplied :class:`Standardizer` object
    """

    def __new__(cls, name: str, μ: np.ndarray, σ2: np.ndarray, stdzr: Standardizer, stdzd=False):
        μ_ = np.asarray(μ)
        σ2_ = np.asarray(σ2)
        assert(μ_.shape == σ2_.shape)

        if stdzd:
            μ_, σ2_ = stdzr.unstdz(name, μ_, σ2_)

        uparray_dtype = np.dtype([('μ', μ_.dtype), ('σ2', σ2_.dtype)])

        uparray_prototype = np.empty(μ_.shape, dtype=uparray_dtype)
        uparray_prototype['μ'] = μ_
        uparray_prototype['σ2'] = σ2_

        uparray = uparray_prototype.view(cls)
        uparray.name = name
        uparray.stdzr = stdzr
        uparray.fields = list(uparray_dtype.fields.keys())

        return uparray

    def __array_finalize__(self, uparray):
        if uparray is None:
            return
        self.name = getattr(uparray, 'name', None)
        self.fields = getattr(uparray, 'fields', None)
        self.stdzr = getattr(uparray, 'stdzr', None)

    @property
    def z(self) -> UncertainArray:
        """Standardized values"""

        zmean, zvar = self.stdzr.stdz(self.name, self.μ, self.σ2)

        return UncertainArray(f'{self.name}_z', zmean, zvar, stdzr=self.stdzr)

    @property
    def t(self) -> UncertainArray:
        """Transformed values"""

        tmean, tvar = self.stdzr.transform(self.name, self.μ, self.σ2)

        return UncertainArray(f'{self.name}_t', tmean, tvar, stdzr=self.stdzr)

    @property
    def _ftransform(self):
        return self.stdzr.transforms.get(self.name, [skip, skip])[0]

    @property
    def _as_uncarray(self):
        return unp.uarray(self.z.μ, self.z.σ)

    def _from_uncarray(self, name, uncarray):
        z = UncertainArray._from_uncarray(name, uncarray)
        return self._from_z(z)

    @property
    def dist(self) -> rv_continuous:
        """Array of :func:`scipy.stats.rv_continuous` objects.

        If the transformation associated with the array's parameter is log/exp, this is a `lognorm` distribution object
        with ``scale=self.μ`` and ``s=self.t.σ``. Otherwise it is a `norm` distribution with ``loc=self.μ`` and
        ``scale=self.σ``. See the scipy documentation on `LogNormal`_ and `Normal`_ random variables for more
        explanation and a list of methods.

        .. _Normal: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
        .. _LogNormal: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html
        """
        dists = {
            skip: super().dist,
            np.log: lognorm(scale=self.μ, s=self.σ),
            logit: LogitNormal(loc=self.μ, scale=self.σ)
        }
        return dists[self._ftransform]

    def sum(self, axis=None, dtype=None, out=None, keepdims=False, **kwargs):
        self._warn_if_poorly_defined()
        kwargs |= dict(axis=axis, dtype=dtype, out=out, keepdims=keepdims)
        z = self.z.sum(**kwargs)
        return self._from_z(z)

    def mean(self, axis=None, dtype=None, out=None, keepdims=False, **kwargs):
        """The natural-space distribution parameters which represent the mean of the transformed-space distributions"""
        kwargs |= dict(axis=axis, dtype=dtype, out=out, keepdims=keepdims)
        z = self.z.mean(**kwargs)
        return self._from_z(z)

    def _from_z(self, z):
        name = z.name.replace('_z', '')
        fields = {dim: z[dim] for dim in z.fields}
        return UncertainParameterArray(name, **fields, stdzr=self.stdzr, stdzd=True)

    def _from_t(self, t):
        name = t.name.replace('_t', '')
        t.μ, t.σ2 = self.stdzr.untransform(name, t.μ, t.σ2)
        return UncertainParameterArray(name, **{dim: t[dim] for dim in t.fields}, stdzr=self.stdzr,
                                       stdzd=False)

    def _warn_if_dissimilar(self, other):
        if isinstance(other, UncertainParameterArray):
            if not self.stdzr == other.stdzr:
                warnings.warn('uparrays have dissimilar Standardizers')

    def _warn_if_poorly_defined(self):
        if self._ftransform is not skip:
            warnings.warn(f'Transform is poorly defined for {self._ftransform}; results may be unexpected.')

    # def __repr__(self):
    #     return f'{self.name}{self.fields}: {np.asarray(self)}'

    def __getitem__(self, item):
        default = super(UncertainArray, self).__getitem__(item)
        if isinstance(item, int) or (isinstance(item, tuple) and all(isinstance(val, int) for val in item)):
            arrays = {name: value for name, value in zip(default.dtype.names, default)}
        elif isinstance(item, slice):
            # arrays = {layer.names[0]: layer.values() for layer in default.as_list()}
            return default
        else:
            return default.view(np.ndarray)
        return UncertainParameterArray(self.name, stdzr=self.stdzr, stdzd=False, **arrays)

    def __add__(self, other):
        self._warn_if_dissimilar(other)
        self._warn_if_poorly_defined()
        if isinstance(other, UncertainParameterArray):
            new = self._from_t(self.t.__add__(other.t))
            new.stdzr = Standardizer(**(self.stdzr | other.stdzr))
        else:
            new = super().__add__(other)
        return new

    def __sub__(self, other):
        self._warn_if_dissimilar(other)
        self._warn_if_poorly_defined()
        if isinstance(other, UncertainParameterArray):
            new = self._from_t(self.t.__sub__(other.t))
            new.stdzr = Standardizer(**(self.stdzr | other.stdzr))
        else:
            new = super().__sub__(other)
        return new

    def __rsub__(self, other):
        self._warn_if_dissimilar(other)
        self._warn_if_poorly_defined()
        if isinstance(other, UncertainParameterArray):
            new = self._from_t(self.t.__rsub__(other.t))
            new.stdzr = Standardizer(**(other.stdzr | self.stdzr))
        else:
            new = super().__rsub__(other)
        return new


class MVUncertainParameterArray(np.ndarray):
    r"""Structured array of multiple parameter means and variances along with correlations.

    This class is essentially a combination of the :class:`ParameterArray` and :class:`UncertainParameterArray` classes.

    See Also
    --------
    `multivariate_normal <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_normal.html>`_:
    scipy `Multivariate Normal` random variable

    :class:`ParameterArray`

    :class:`UncertainParameterArray`

    Parameters
    ----------
    \*uparrays: UncertainParameterArray
        UParrays of identical shape containing the marginal mean and variance of each parameter
    stdzr : Standardizer
        An instance  of :class:`Standardizer`
    stdzd : bool, default False
        Whether the supplied values are on standardized scale instead of the natural scale

    Examples
    --------
    Create an MVUParray from two UParrays with negative correlation

    >>> import numpy as np
    >>> from gumbi import uparray, mvuparray, Standardizer
    >>>
    >>> stdzr = Standardizer(m = {'μ': -5.30, 'σ': 0.582}, r = {'μ': -0.307, 'σ': 0.158}, log_vars=['d', 'c'])
    >>> m_upa = uparray('c', np.arange(1,5)/10, np.arange(1,5)/100, stdzr)
    >>> r_upa = uparray('d', np.arange(1,5)/10+0.5, np.arange(1,5)/100*2, stdzr)
    >>> cor = np.array([[1, -0.6], [-0.6, 1]])
    >>> cor
    array([[ 1. , -0.6],
           [-0.6,  1. ]])
    >>> mvup = mvuparray(m_upa, r_upa, cor=cor)

    The MVUParray is displayed as an array of tuples ((μ_a, μ_b), (σ2_a, σ2_b))

    >>> mvup
    ('c', 'd')['μ', 'σ2']: [((0.1, 0.6), (0.01, 0.02)) ((0.2, 0.7), (0.02, 0.04))
                            ((0.3, 0.8), (0.03, 0.06)) ((0.4, 0.9), (0.04, 0.08))]

    The marginal means or variances can be extracted as ParameterArrays:

    >>> mvup.μ
    ('c', 'd'): [(0.1, 0.6) (0.2, 0.7) (0.3, 0.8) (0.4, 0.9)]

    The component UncertainParameterArrays can be extracted with the :meth:`get` method:

    >>> mvup.get('d')
    r['μ', 'σ2']: [(0.6, 0.02) (0.7, 0.04) (0.8, 0.06) (0.9, 0.08)]

    Slicing and indexing works as normal:

    >>> mvup[::2]
    ('c', 'd'): [((0.1, 0.6), (0.01, 0.02)) ((0.3, 0.8), (0.03, 0.06))]

    Transformed and standardized distributions can be obtained as with UncertainParameterArray

    >>> mvup.t
    ('m_t', 'r_t')['μ', 'σ2']: [((-2.30258509, -0.51082562), (0.01, 0.02))
                                ((-1.60943791, -0.35667494), (0.02, 0.04))
                                ((-1.2039728 , -0.22314355), (0.03, 0.06))
                                ((-0.91629073, -0.10536052), (0.04, 0.08))]
    >>> mvup.z
    ('m_z', 'r_z')['μ', 'σ2']: [((5.15019743, -1.29003559), (0.02952256, 0.80115366))
                                ((6.34117197, -0.31439838), (0.05904512, 1.60230732))
                                ((7.03784742,  0.53073702), (0.08856768, 2.40346098))
                                ((7.53214651,  1.27619927), (0.11809024, 3.20461465))]

    For 0-d MVUParrays (or individual elements of larger arrays), the :meth:`dist` exposes the scipy
    `multivariate_normal` object. Because this distribution is defined in standardized space (by default, `stdzd=True`)
    or transformed spaced (`stdzd=False`), ParameterArrays may be the most convenient way to pass arguments to the
    distribution methods:

    >>> pa = mvup.parray(m=0.09, d=0.61)
    >>> mvup[0].dist().cdf(pa.z.values())
    0.023900979112885523
    >>> mvup[0].dist(stdzd=False).cdf(pa.t.values())
    0.023900979112885523

    Perhaps most importantly, the :meth:`rvs` allows drawing correlated samples from the joint distribution, returned as
    a ParameterArray:

    >>> mvup[0].dist.rvs(10, random_state=2021)
    ('c', 'd'): [(0.0962634 , 0.74183363) (0.09627651, 0.56437764)
                 (0.09140986, 0.64790721) (0.09816149, 0.70518567)
                 (0.10271404, 0.60974628) (0.09288982, 0.60933939)
                 (0.0983131 , 0.63588871) (0.12262933, 0.45941758)
                 (0.1070759 , 0.4918009 ) (0.11118635, 0.49708401)]
    >>> mvup[0].dist.rvs(1, random_state=2021).z
    ('m_z', 'r_z'): (5.08476443, 0.05297293)

    Allowing us to easily visualize the joint distribution:

    >>> import pandas as pd
    >>> import seaborn as sns
    >>> sns.jointplot(x='c', y='d', data=pd.DataFrame(mvup[0].dist.rvs(1000).as_dict()), kind='kde')

    """

    def __new__(cls, *uparrays, cor, stdzr=None):

        shape = uparrays[0].shape
        assert all([upa.shape == shape for upa in uparrays])
        assert cor.shape[0] == len(uparrays)
        stdzr = uparrays[0].stdzr if stdzr is None else stdzr

        μ_ = ParameterArray(**{upa.name: upa.μ for upa in uparrays}, stdzr=stdzr)
        σ2_ = ParameterArray(**{upa.name: upa.σ2 for upa in uparrays}, stdzr=stdzr)

        mvuparray_dtype = np.dtype([('μ', μ_.dtype), ('σ2', σ2_.dtype)])
        mvuparray_prototype = np.empty(shape, dtype=mvuparray_dtype)
        mvuparray_prototype['μ'] = μ_
        mvuparray_prototype['σ2'] = σ2_

        mvuparray = mvuparray_prototype.view(cls)
        mvuparray.names = [upa.name for upa in uparrays]
        mvuparray.stdzr = stdzr
        mvuparray.fields = list(mvuparray_dtype.fields.keys())
        mvuparray.cor = cor

        return mvuparray

    def __array_finalize__(self, mvup):
        if mvup is None:
            return
        self.names = getattr(mvup, 'names', None)
        self.fields = getattr(mvup, 'fields', None)
        self.stdzr = getattr(mvup, 'stdzr', None)
        self.cor = getattr(mvup, 'cor', None)

    def __repr__(self):
        return f'{tuple(self.names)}{self.fields}: {np.asarray(self)}'

    def __getitem__(self, item):
        default = super().__getitem__(item)
        if isinstance(item, int) or (isinstance(item, tuple) and all(isinstance(val, int) for val in item)):
            arrays = [self.get(name)[item] for name in self.names]
        elif isinstance(item, slice):
            # arrays = [self.get(name)[item] for name in self.names]
            # Not sure why this "just works" for slices...
            return default
        else:
            return default.view(ParameterArray)
        return self.mvuparray(*arrays)

    def get(self, name, default=None) -> UncertainParameterArray | MVUncertainParameterArray:
        """Return one component parameter as an UncertainParameterArray or a subset as an MVUncertainParameterArray"""
        if isinstance(name, str):
            if name in self.names:
                return self.uparray(name, self['μ'][name].values(), self['σ2'][name].values())
            else:
                return default
        elif isinstance(name, list):
            idxs = [self.names.index(n) for n in name]
            return self.mvuparray([self.get(n) for n in name], cor=self.cor[idxs, :][:, idxs])

    @property
    def μ(self) -> ParameterArray:
        """Means"""
        return self['μ']

    @μ.setter
    def μ(self, val):
        self['μ'] = val

    @property
    def σ2(self) -> ParameterArray:
        """Marginal variances"""
        return self['σ2']

    @σ2.setter
    def σ2(self, val):
        self['σ2'] = val

    @property
    def σ(self) -> ParameterArray:
        """Standard deviations"""
        return self.parray(**{k: np.sqrt(v) for k, v in self['σ2'].as_dict().items()})

    @property
    def t(self) -> MVUncertainParameterArray:
        """Transformed values.

        Returns a MVUncertainParameterArray with the same Standardizer values as the current instance but with all
        transforms set to `lambda x: x`.
        """
        stdzr = Standardizer(**{k+'_t': v for k, v in self.stdzr.items()})
        return self.mvuparray(*[self.get(name).t for name in self.names], stdzr=stdzr)

    @property
    def z(self) -> MVUncertainParameterArray:
        """Standardized values.

        Returns a MVUncertainParameterArray with a default Standardizer (mean and SD of all variables set to zero and
        all transforms set to `lambda x: x`).
        """
        stdzr = Standardizer(**{k+'_z': {'μ': 0, 'σ2': 1} for k in self.names})
        return self.mvuparray(*[self.get(name).z for name in self.names], stdzr=stdzr)

    def parray(self, *args, **kwargs) -> ParameterArray:
        """Create a ParameterArray using this instance's Standardizer"""
        kwargs.setdefault('stdzr', self.stdzr)
        return ParameterArray(*args, **kwargs)

    def uparray(self, *args, **kwargs) -> UncertainParameterArray:
        """Create an UncertainParameterArray using this instance's Standardizer"""
        kwargs.setdefault('stdzr', self.stdzr)
        return UncertainParameterArray(*args, **kwargs)

    def mvuparray(self, *args, **kwargs) -> MVUncertainParameterArray:
        """Create an MVUncertainParameterArray using this instance's Standardizer"""
        kwargs.setdefault('stdzr', self.stdzr)
        kwargs.setdefault('cor', self.cor)
        return MVUncertainParameterArray(*args, **kwargs)

    def cov(self, stdzd=True, whiten=1e-10):
        """Covariance matrix (only supported for 0-D MVUParrays)"""
        # TODO: numpy versions > 1.19.3 can have bizarre inscrutable errors when handling mvup.cov. Monitor for fixes.
        if np.__version__ > '1.19.3':
            warnings.warn('numpy version >1.19.3 may lead to inscrutable linear algebra errors with mvup.cov. May just be on Windows/WSL. Hopefully fixed soon.')
        if self.ndim != 0:
            raise NotImplementedError('Multidimensional multivariate covariance calculations are not yet supported.')

        σ = self.z.σ.values() if stdzd else self.t.σ.values()

        cov = np.diag(σ) @ self.cor @ np.diag(σ)

        if whiten:
            cov += whiten*np.eye(*cov.shape)

        return cov

    @property
    def dist(self) -> MultivariateNormalish:
        """Scipy :func:`multivariate_normal` object (only supported for 0-D MVUParrays)"""
        # TODO: numpy versions > 1.19.3 can have bizarre inscrutable errors when handling mvup.dist. Monitor for fixes.
        if np.__version__ > '1.19.3':
            warnings.warn('numpy version >1.19.3 may lead to inscrutable linear algebra errors with mvup.dist. May just be on Windows/WSL. Hopefully fixed soon.')
        if self.ndim != 0:
            raise NotImplementedError('Multidimensional multivariate distributions are not yet supported.')
        return MultivariateNormalish(mean=self.μ, cov=self.cov(stdzd=True))

    def mahalanobis(self, parray: ParameterArray) -> float:
        """Calculates the Mahalanobis distance between the MVUParray distribution and a (ParameterArray) point."""
        cov_inv = np.linalg.inv(self.cov(stdzd=True))
        points = np.stack([parray.z.get(p+'_z').values() for p in self.names])
        μ = np.stack([self.z.μ.get(p+'_z').values() for p in self.names])
        diff = points-μ
        return np.sqrt(diff.T @ cov_inv @ diff)

    def outlier_pval(self, parray: ParameterArray) -> float:
        """Calculates the p-value that a given (ParameterArray) point is an outlier from the MVUParray distribution."""
        MD = self.mahalanobis(parray)
        n_params = len(self.names)
        pval = 1-chi2.cdf(MD**2, df=n_params)
        return pval

