from __future__ import annotations  # Necessary for self-type annotations until Python >3.10

import pickle
import warnings
from collections import namedtuple
from dataclasses import dataclass
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from scipy.special import logit, expit

from .utils import skip

__all__ = ['Standardizer', 'TidyData', 'WideData', 'DataSet']


class Standardizer(dict):
    r"""Container for dict of mean (μ) and variance (σ2) for every parameter.

    :class:`Standardizer` objects allow transformation and normalization of datasets. The main methods are :meth:`stdz`,
    which attempts to coerce the values of a given variable to a standard normal distribution (`z-scores`), and its
    complement :meth:`unstdz`. The steps are

    .. math::
        \mathbf{\text{tidy}} \rightarrow \text{transform} \rightarrow \text{mean-center} \rightarrow \text{scale}
        \rightarrow \mathbf{\text{tidy.z}}

    For example, reaction `rate` must clearly be strictly positive, so we use a `log` transformation so that it behaves
    as a normally-distributed random variable. We then mean-center and scale this transformed value to obtain `z-scores`
    indicating how similar a given estimate is to all the other estimates we've observed. `Standardizer` stores the
    transforms and population mean and variance for every parameter, allowing us to convert back and forth
    between natural space (:math:`rate`), transformed space (:math:`\text{ln}\; rate`), and standardized space
    (:math:`\left( \text{ln}\; rate  - \mu_{\text{ln}\; rate} \right)/\sigma_{\text{ln}\; rate}`).

    Typically, a :class:`Standardizer` will be constructed from a dataframe (:meth:`from_DataFrame`),
    but the individual means and variances can be provided at instantiation as well. Note, however,
    that these should be the mean/std of the *transformed* variable. For example, if `r` should be treated as
    log-normal with a natural-space mean of 1 and variance of 0.1, the right way to instantiate the class
    would be `Standardizer(d={'μ': 0, 'σ2': 0.1}, log_vars=['d'])`.


    Notes
    -----
    :class:`Standardizer` is just a `dictionary <https://docs.python.org/3/tutorial/datastructures.html#dictionaries>`_
    with some extra methods and defaults, so standard dictionary methods like :meth:`dict.update` still work.


    Parameters
    ----------
    log_vars: list, optional
        List of input and output variables to be treated as log-normal.
    logit_vars: list, optional
        List of input and output variables to be treated as logit-normal.
    **kwargs
        Mean and variance of each variable as a dictionary, e.g. d={'μ': 0, 'σ2': 0.1}


    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from gumbi import Standardizer
    >>> stdzr = Standardizer(x={'μ': 1, 'σ2': 0.1}, d={'μ': 0, 'σ2': 0.1}, log_vars=['d'])

    Transforming and standardizing a single parameter:

    >>> stdzr.transform('x', μ=1)
    1
    >>> stdzr.stdz('x', 1)
    0.0
    >>> stdzr.unstdz('x', 0)
    1.0
    >>> stdzr.stdz('x', 1+0.1**0.5)
    1.0  # approximately
    >>> stdzr.unstdz('x', 1)
    1.316227766016838
    >>> stdzr.stdz('d', 1)
    0.0
    >>> stdzr.stdz('d', np.exp(0.1**0.5))
    1.0  # approximately

    Transforming and standardizing a distribution:

    >>> stdzr.transform('x', μ=1., σ2=0.1)
    (1, 0.1)
    >>> stdzr.stdz('x', 1, 0.1)
    (0.0, 1.0)
    >>> stdzr.stdz('d', 1, 0.1)
    (0.0, 1.0)
    >>> stdzr.transform('d', 1, 0.1)
    (0.0, 0.1)

    Standardizing a series:

    >>> x_series = pd.Series(np.arange(1,5), name='x')
    >>> stdzr.stdz(x_series)
    0    0.000000
    1    3.162278
    2    6.324555
    3    9.486833
    Name: x, dtype: float64
    >>> r_series = pd.Series(np.arange(1,5), name='d')
    >>> stdzr.stdz(r_series)
    0    0.000000
    1    2.191924
    2    3.474117
    3    4.383848
    Name: r, dtype: float64

    """

    # TODO: Standardizer: make transform suggestions based on provided tidy? e.g., all>0 -> log/exp

    def __init__(self, log_vars=None, logit_vars=None, **kwargs):
        self.validate(kwargs)
        for name, stats in kwargs.items():
            if 'σ2' not in stats:
                stats['σ2'] = stats['σ']**2
                del stats['σ']
        super().__init__(**kwargs)
        self._transforms = {var: [skip, skip] for var in kwargs.keys()}
        if log_vars is not None:
            log_vars = [log_vars] if isinstance(log_vars, str) else log_vars
            if not isinstance(log_vars, list):
                raise TypeError('log_vars must be a list or str')
            self._transforms |= {var: [np.log, np.exp] for var in log_vars}
        if logit_vars is not None:
            logit_vars = [logit_vars] if isinstance(logit_vars, str) else logit_vars
            if not isinstance(logit_vars, list):
                raise TypeError('logit_vars must be a list or str')
            self._transforms |= {var: [logit, expit] for var in logit_vars}
        self._log_vars = log_vars if log_vars is not None else []
        self._logit_vars = logit_vars if logit_vars is not None else []

    def __or__(self, __dct) -> Standardizer:
        new_dct = super().__or__(__dct)
        stdzr = Standardizer(**new_dct)
        if isinstance(__dct, Standardizer):
            stdzr.transforms = self.transforms | __dct.transforms
        else:
            stdzr.transforms = self.transforms
        return stdzr

    def __ror__(self, __dct) -> Standardizer:
        new_dct = super().__ror__(__dct)
        stdzr = Standardizer(**new_dct)
        stdzr.transforms = self.transforms
        return stdzr

    def __repr__(self):
        summary = '\n\t'.join([
            f'Standardizer:',
            f'log_vars: {self.log_vars}',
            f'logit_vars: {self.logit_vars}',
        ]) + '\n\n' + str({**self})
        return summary

    @property
    def log_vars(self) -> list[str]:
        """List of log-normal variables"""
        return self._log_vars

    @log_vars.setter
    def log_vars(self, var_list):
        var_list = [var_list] if isinstance(var_list, str) else var_list
        if not isinstance(var_list, list):
            raise TypeError('log_vars must be a list or str')
        self._log_vars = var_list
        self._transforms |= {var: [np.log, np.exp] for var in var_list}

    @property
    def logit_vars(self) -> list[str]:
        """List of logit-normal variables"""
        return self._logit_vars

    @logit_vars.setter
    def logit_vars(self, var_list):
        var_list = [var_list] if isinstance(var_list, str) else var_list
        if not isinstance(var_list, list):
            raise TypeError('logit_vars must be a list or str')
        self._logit_vars = var_list
        self._transforms |= {var: [logit, expit] for var in var_list}

    @property
    def transforms(self) -> dict:
        """Collection of forward and reverse transform functions for each variable"""
        return self._transforms

    @transforms.setter
    def transforms(self, dct) -> dict:
        self._transforms = dct
        self._log_vars = [v for v, lst in dct.items() if lst[0] is np.log]
        self._logit_vars = [v for v, lst in dct.items() if lst[0] is logit]

    @classmethod
    def validate(cls, dct: dict):
        """Ensures provided dictionary has all required attributes"""
        assert all('μ' in sub.keys() for sub in dct.values())
        assert all(('σ' in sub.keys() or 'σ2' in sub.keys()) for sub in dct.values())

    @classmethod
    def from_DataFrame(cls, df: pd.DataFrame, log_vars=None, logit_vars=None):
        """Construct from wide-form DataFrame"""
        float_columns = df.dtypes[df.dtypes == 'float64'].index.to_list()

        new = cls(log_vars=log_vars, logit_vars=logit_vars)

        dct = (df[float_columns]
               .apply(new.transform)
               .agg([np.mean, np.var])
               .rename(index={"mean": "μ", "var": "σ2"})
               .to_dict()
               )

        return new | dct

    def transform(self, name: str | pd.Series, μ: float = None, σ2: float = None) -> float | tuple | pd.Series:
        """Transforms a parameter, distribution, or Series

        Parameters
        ----------
        name: str or pd.Series
            Name of parameter. If a Series is supplied, the name of the series must be the parameter name.
        μ: float, optional
            Value of parameter or mean of parameter distribution. Only optional if first argument is a Series.
        σ2: float, optional
            Variance of parameter distribution.

        Returns
        -------
        float, tuple, or pd.Series
            Transformed parameter, (mean, variance) of untransformed distribution, or untransformed Series
        """
        if isinstance(series := name, pd.Series):
            return self._transform_value(series.name, series)
        elif μ is None:
            raise ValueError('μ cannot be None')
        if σ2 is None:
            return self._transform_value(name, μ)
        else:
            return self._transform_dist(name, μ, σ2)

    def untransform(self, name: str | pd.Series, μ: float = None, σ2: float = None) -> float | tuple | pd.Series:
        """Untransforms a parameter, distribution, or Series

        Parameters
        ----------
        name: str or pd.Series
            Name of parameter. If a Series is supplied, the name of the series must be the parameter name.
        μ: float, optional
            Value of parameter or mean of parameter distribution. Only optional if first argument is a Series.
        σ2: float, optional
            Variance of parameter distribution.

        Returns
        -------
        float, tuple, or pd.Series
            Untransformed parameter, (mean, variance) of untransformed distribution, or untransformed Series
        """
        if isinstance(series := name, pd.Series):
            return self._untransform_value(series.name, series)
        if σ2 is None:
            return self._untransform_value(name, μ)
        else:
            return self._untransform_dist(name, μ, σ2)

    def stdz(self, name: str | pd.Series, μ: float = None, σ2: float = None) -> float | tuple | pd.Series:
        """Transforms, mean-centers, and scales a parameter, distribution, or Series

        Parameters
        ----------
        name: str or pd.Series
            Name of parameter. If a Series is supplied, the name of the series must be the parameter name.
        μ: float, optional
            Value of parameter or mean of parameter distribution. Only optional if first argument is a Series.
        σ2: float, optional
            Variance of parameter distribution.

        Returns
        -------
        float, tuple, or pd.Series
            Standardized parameter, (mean, variance) of standardized distribution, or standardized Series
        """

        if isinstance(series := name, pd.Series):
            return self._stdz_value(series.name, series)
        if σ2 is None:
            return self._stdz_value(name, μ)
        else:
            return self._stdz_dist(name, μ, σ2)

    def unstdz(self, name: str | pd.Series, μ: float = None, σ2: float = None) -> float | tuple | pd.Series:
        """Untransforms, un-centers, and un-scales a parameter, distribution, or Series

        Parameters
        ----------
        name: str or pd.Series
            Name of parameter. If a Series is supplied, the name of the series must be the parameter name.
        μ: float, optional
            Value of parameter or mean of parameter distribution. Only optional if first argument is a Series.
        σ2: float, optional
            Variance of parameter distribution.

        Returns
        -------
        float, tuple, or pd.Series
            Unstandardized parameter, (mean, variance) of unstandardized distribution, or unstandardized Series
        """

        if isinstance(series := name, pd.Series):
            return self._unstdz_value(series.name, series)
        if σ2 is None:
            return self._unstdz_value(name, μ)
        else:
            return self._unstdz_dist(name, μ, σ2)

    def _transform_value(self, name: str, x: float) -> float:
        ftransform = self.transforms.get(name, [skip, skip])[0]

        return ftransform(x)

    def _untransform_value(self, name: str, x: float) -> float:
        rtransform = self.transforms.get(name, [skip, skip])[1]
        x_ = rtransform(x)

        return x_

    def _stdz_value(self, name: str, x: float) -> float:
        x_ = self.transform(name, x)
        μ = self.get(name, {'μ': 0})['μ']
        σ2 = self.get(name, {'σ2': 1})['σ2']
        σ = np.sqrt(σ2)
        return (x_ - μ) / σ

    def _unstdz_value(self, name: str, z: float) -> float:
        μ = self.get(name, {'μ': 0})['μ']
        σ2 = self.get(name, {'σ2': 1})['σ2']
        σ = np.sqrt(σ2)
        x_ = z * σ + μ
        return self.untransform(name, x_)

    @property
    def mean_transforms(self):
        """Function that transforms the mean of a distribution.

        These transform's should follow scipy's conventions such that a distribution can be defined in the given
        space by passing (loc=μ, scale=σ2**0.5). For a lognormal variable, an RV defined as ``lognorm(loc=μ,
        scale=σ2**0.5)`` in "natural" space is equivalent to ``norm(loc=np.log(μ), scale=σ2**0.5)`` in log space,
        so this transform should return ``np.log(μ)`` when converting from natural to log space, and ``np.exp(μ)``
        when converting from log to natural space. Similarly for a logit-normal variable, an RV defined as
        ``logitnorm(loc=μ, scale=σ2**0.5))`` in natural space is equivalent to ``norm(loc=logit(μ), scale=σ2**0.5)``
        in logit space, so this transform should return ``logit(μ)`` when converting from natural to logit space,
        and ``expit(μ)`` when converting from logit to natural space.
        """

        # Forward and reverse transform for each variable type
        transforms = {skip: [lambda μ, σ2: μ,
                             lambda μ, σ2: μ],
                      # Note these are no longer strictly mean and variance. They are defined to be compatible with
                      # scipy.stats.lognormal definition
                      np.log: [lambda μ, σ2: np.log(μ),
                               lambda μ, σ2: np.exp(μ)],
                      logit: [lambda μ, σ2: logit(μ),
                              lambda μ, σ2: expit(μ)]
                      }
        return transforms

    @property
    def var_transforms(self):
        """Function that transforms the variance of a distribution.

        These transform's should follow scipy's conventions such that a distribution can be defined in the given
        space by passing (loc=μ, scale=σ2**0.5). Accordingly, since both log-normal and logit-normal variables are
        defined in terms of the scale (standard deviation) in their respective transformed spaces, this function
        simply returns the variance unchanged in these cases.
        """

        # Forward and reverse transform for each variable type
        transforms = {skip: [lambda μ, σ2: σ2,
                             lambda μ, σ2: σ2],
                      np.log: [lambda μ, σ2: σ2,
                               lambda μ, σ2: σ2],
                      logit: [lambda μ, σ2: σ2,
                              lambda μ, σ2: σ2]
                      }
        return transforms

    def _transform_dist(self, name: str, mean: float, var: float) -> tuple:
        f_transform = self.transforms.get(name, [skip, skip])[0]
        f_mean_transform = self.mean_transforms[f_transform][0]
        f_var_transform = self.var_transforms[f_transform][0]

        mean_ = f_mean_transform(mean, var)
        var_ = f_var_transform(mean, var)
        return mean_, var_

    def _untransform_dist(self, name: str, mean: float, var: float) -> tuple:
        f_transform = self.transforms.get(name, [skip, skip])[0]
        r_mean_transform = self.mean_transforms[f_transform][1]
        r_var_transform = self.var_transforms[f_transform][1]

        mean_ = r_mean_transform(mean, var)
        var_ = r_var_transform(mean, var)

        return mean_, var_

    def _stdz_dist(self, name: str, mean: float, var: float) -> tuple:
        mean_, var_ = self.transform(name, mean, var)
        μ = self.get(name, {'μ': 0})['μ']
        σ2 = self.get(name, {'σ2': 1})['σ2']
        σ = np.sqrt(σ2)
        mean_z = (mean_ - μ) / σ
        var_z = var_ / σ2
        return mean_z, var_z

    def _unstdz_dist(self, name: str, z_mean: float, z_var: float) -> tuple:
        μ = self.get(name, {'μ': 0})['μ']
        σ2 = self.get(name, {'σ2': 1})['σ2']
        σ = np.sqrt(σ2)
        mean_ = z_mean * σ + μ
        var_ = z_var * σ2
        mean, var = self.untransform(name, mean_, var_)
        return mean, var


@dataclass
class MetaFrame(pd.DataFrame, ABC):
    """Abstract Base Class for :class:`WideData` and :class:`TidyData`."""

    df: pd.DataFrame
    outputs: list
    log_vars: list = None
    logit_vars: list = None
    names_column: str = 'Variable'
    values_column: str = 'Value'
    stdzr: Standardizer = None

    _metadata = ['df', 'outputs', 'log_vars', 'logit_vars', 'names_column', 'values_column', 'stdzr']

    def __post_init__(self):
        super(MetaFrame, self).__init__(self.df)
        if self.stdzr is None:
            self.stdzr = Standardizer.from_DataFrame(self.df, log_vars=self.log_vars, logit_vars=self.logit_vars)
        else:
            self.log_vars = self.stdzr.log_vars
            self.logit_vars = self.stdzr.logit_vars
        del self.df

    def __repr__(self):
        cls = self.__class__.__name__
        df_repr = super(MetaFrame, self).__repr__()

        summary = '\n\t'.join([
            f'{cls}:',
            f'outputs: {self.outputs}',
            f'inputs: {self.inputs}',
        ]) + '\n\n' + df_repr
        return summary

    @property
    @abstractmethod
    def z(self) -> pd.DataFrame:
        """Standardized data values."""
        pass

    @property
    @abstractmethod
    def t(self) -> pd.DataFrame:
        """Transformed data values."""
        pass

    @property
    def specs(self) -> dict:
        """Provides keyword arguments for easy instantiation of a similar object."""
        return dict(outputs=self.outputs, names_column=self.names_column, values_column=self.values_column,
                    stdzr=self.stdzr, log_vars=self.log_vars, logit_vars=self.logit_vars)

    @property
    def inputs(self) -> list[str]:
        """Columns of dataframe not contained in :attr:`outputs`."""
        return [col for col in self.columns if col not in self.outputs]

    @property
    def float_inputs(self) -> list[str]:
        """Columns of dataframe with "float64" dtype."""
        return [col for col in self.inputs if self[col].dtype == 'float64']

    @classmethod
    def _wide_to_tidy_(cls, wide, outputs, names_column='Variable', values_column='Value'):
        inputs = [col for col in wide.columns if col not in outputs]
        tidy = wide.melt(id_vars=inputs, value_vars=outputs, var_name=names_column,
                         value_name=values_column)
        return tidy

    @classmethod
    def _tidy_to_wide_(cls, tidy, names_column='Variable', values_column='Value'):
        inputs = [col for col in tidy.columns if col not in [names_column, values_column]]
        wide = (tidy
                .pivot(index=inputs, columns=names_column, values=values_column)
                .reset_index()
                .rename_axis(columns=None)
                )
        return wide


class WideData(MetaFrame):
    """Container for wide-form tabular data, allowing simple access to standardized and/or transformed values.

    Note that :class:`WideData` is instantiated with a **wide-form** dataframe. This class is not intended to be
    instantiated directly, use :class:`DataSet` instead. :class:`WideData` subclasses pandas' DataFrame,
    which everyone says is a bad idea, so be prepared for unexpected behavior if instantiated directly. Namely, in-place
    modifications return a :class:`WideData` type correctly, but slices return a `pd.DataFrame` type.

    Parameters
    ----------
    data: pd.DataFrame
        A wide-form dataframe.
    outputs: list
        Columns of `data` to be treated as outputs.
    names_column: str, default 'Variable'
        Name to be used in tidy view for column containing output names.
    values_column: str, default 'Value'
        Name to be used in tidy view for column containing output values.
    log_vars: list, optional
        List of input and output variables to be treated as log-normal. Ignored if `stdzr` is supplied.
    logit_vars: list, optional
        List of input and output variables to be treated as logit-normal. Ignored if `stdzr` is supplied.
    stdzr: Standardizer, optional
        An :class:`Standardizer` instance. If not supplied, one will be created automatically.
    """

    @property
    def z(self) -> pd.DataFrame:
        """Standardized data values."""
        df_ = self.copy()
        cols = self.outputs + self.float_inputs
        df_[cols] = df_[cols].apply(self.stdzr.stdz)
        return df_

    @property
    def t(self) -> pd.DataFrame:
        """Transformed data values."""
        df_ = self.copy()
        cols = self.outputs + self.float_inputs
        df_[cols] = df_[cols].apply(self.stdzr.transform)
        return df_

    def to_tidy(self) -> TidyData:
        """Converts to TidyData"""
        tidy = TidyData(self, **self.specs)
        return tidy

    @classmethod
    def from_tidy(cls, tidy, outputs=None, names_column='Variable', values_column='Value',
                  stdzr=None, log_vars=None, logit_vars=None):
        """Constructs `WideData` from a tidy-form dataframe. See :class:`WideData` for explanation of arguments."""
        outputs = outputs if outputs is not None else list(tidy[names_column].unique())
        wide = cls._tidy_to_wide_(tidy, names_column=names_column, values_column=values_column)
        return cls(wide, outputs=outputs, names_column=names_column, values_column=values_column,
                   stdzr=stdzr, log_vars=log_vars, logit_vars=logit_vars)


class TidyData(MetaFrame):
    """Container for tidy-form tabular data, allowing simple access to standardized and/or transformed values.

    Note that :class:`TidyData` is instantiated with a **wide-form** dataframe. This class is not intended to be
    instantiated directly, use :class:`DataSet` instead. :class:`TidyData` subclasses pandas' DataFrame,
    which everyone says is a bad idea, so be prepared for unexpected behavior if instantiated directly. Namely, in-place
    modifications return a :class:`TidyData` type correctly, but slices return a `pd.DataFrame` type.

    Parameters
    ----------
    data: pd.DataFrame
        A wide-form dataframe.
    outputs: list
        Columns of `data` to be treated as outputs.
    names_column: str, default 'Variable'
        Name to be used in tidy view for column containing output names.
    values_column: str, default 'Value'
        Name to be used in tidy view for column containing output values.
    log_vars: list, optional
        List of input and output variables to be treated as log-normal. Ignored if `stdzr` is supplied.
    logit_vars: list, optional
        List of input and output variables to be treated as logit-normal. Ignored if `stdzr` is supplied.
    stdzr: Standardizer, optional
        An :class:`Standardizer` instance. If not supplied, one will be created automatically.
    """

    def __post_init__(self):
        tidy = self._wide_to_tidy_(self.df, outputs=self.outputs, names_column=self.names_column,
                                   values_column=self.values_column)
        self.df = tidy
        super(TidyData, self).__post_init__()

    @property
    def z(self) -> pd.DataFrame:
        """Standardized data values."""
        wide = self.to_wide()
        specs = dict(outputs=self.outputs, names_column=self.names_column, values_column=self.values_column)
        wd = WideData(wide, **specs, stdzr=self.stdzr)
        z = self._wide_to_tidy_(wd.z, **specs)
        return z

    @property
    def t(self) -> pd.DataFrame:
        """Transformed data values."""
        wide = self.to_wide()
        specs = dict(outputs=self.outputs, names_column=self.names_column, values_column=self.values_column)
        wd = WideData(wide, **specs, stdzr=self.stdzr)
        z = self._wide_to_tidy_(wd.t, **specs)
        return z

    def to_wide(self) -> WideData:
        """Converts to WideData"""
        wide_df = self._tidy_to_wide_(self, names_column=self.names_column, values_column=self.values_column)
        wide = WideData(wide_df, **self.specs)
        return wide


@dataclass
class DataSet:
    """Container for tabular data, allowing simple access to standardized values and wide or tidy dataframe formats.

    :class:`DataSet` is instantiated with a **wide-form** dataframe, with all outputs of a given observation in a
    single row, but allows easy access to the corresponding **tidy** dataframe, with each output in a separate row (
    the :meth:`from_tidy` also allows construction from tidy data`). The titles of the tidy-form columns for the
    output names and their values are supplied at instantiation, defaulting to "Variable" and "Value". For example,
    say we have an observation at position (x,y) with measurements of i, j, and k. The wide-form dataframe would have
    one column for each of x, y, i, j, and k, while the tidy-form dataframe would have a column for each of x and y,
    a "Variable" column where each row contains either "i", "j", or "k" as strings, and a "Value" column containing
    the corresponding measurement. Wide data is more space-efficient and perhaps more intuitive to construct and
    inspect, while tidy data more clearly distinguishes inputs and outputs. These views are accessible through the
    :attr:`wide` and :attr:`tidy` attributes as instances of :class:`WideData` and :class:`TidyData`, respectively.

    As a container for :class:`WideData` and :class:`TidyData`, this class also provides simple access to
    standardized values of the data through `wide.z` and `tidy.z` or transformed values through `wide.t` and
    `tidy.t`. A :class:`Standardizer` instance can be supplied as a keyword argument, otherwise one will be
    constructed automatically from the supplied dataframe with the supplied values of `log_vars` and `logit_vars`.
    Unlike :class:`WideData` and :class:`TidyData`, the :attr:`wide` and :attr:`tidy` attributes of a *DataSet* can
    be altered and sliced while retaining their functionality, with a cursory integrity check. The
    :class:`Standardizer` instance can be updated with :meth:`update_stdzr`, for example following manipulation of
    the data or alteration of :attr:`log_vars` and :attr:`logit_vars`.

    Parameters
    ----------
    data: pd.DataFrame
        A wide-form dataframe. See class method :meth:`from_tidy` for instantiation from tidy data.
    outputs: list
        Columns of `data` to be treated as outputs.
    names_column: str, default 'Variable'
        Name to be used in tidy view for column containing output names.
    values_column: str, default 'Value'
        Name to be used in tidy view for column containing output values.
    log_vars: list, optional
        List of input and output variables to be treated as log-normal. Ignored if `stdzr` is supplied.
    logit_vars: list, optional
        List of input and output variables to be treated as logit-normal. Ignored if `stdzr` is supplied.
    stdzr: Standardizer, optional
        An :class:`Standardizer` instance. If not supplied, one will be created automatically.

    Examples
    --------
    >>> df = pd.read_pickle(test_data / 'estimates_test_data.pkl')
    >>> ds = DataSet.from_tidy(df, names_column='Parameter', log_vars=['Y', 'c', 'b'], logit_vars=['X', 'e'])
    >>> ds
    DataSet:
        wide: [66 rows x 13 columns]
        tidy: [396 rows x 9 columns]
        outputs: ['e', 'f', 'b', 'c', 'a', 'd']
        inputs: ['Code', 'Target', 'Y', 'X', 'Reaction', 'lg10_Z', 'Metric']

    >>> ds.wide = ds.wide.drop(range(0,42,2))
    DataSet:
        wide: [45 rows x 13 columns]
        tidy: [270 rows x 9 columns]
        outputs: ['e', 'f', 'b', 'c', 'a', 'd']
        inputs: ['Code', 'Target', 'Y', 'X', 'Reaction', 'lg10_Z', 'Metric']

    >>> ds.tidy.z  # tidy-form dataframe with standardized values
    >>> ds.wide.z  # wide-form dataframe with standardized values
    """

    data: pd.DataFrame
    outputs: list
    names_column: str = 'Variable'
    values_column: str = 'Value'
    log_vars: list = None
    logit_vars: list = None
    stdzr: Standardizer = None

    def __post_init__(self):
        if self.stdzr is None:
            self.stdzr = Standardizer.from_DataFrame(self.wide, log_vars=self.log_vars, logit_vars=self.logit_vars)
        else:
            self.log_vars = self.stdzr.log_vars
            self.logit_vars = self.stdzr.logit_vars

    def __repr__(self):
        wide_shape = '[{0} rows x {1} columns]'.format(*self.wide.shape)
        tidy_shape = '[{0} rows x {1} columns]'.format(*self.tidy.shape)
        summary = '\n\t'.join([
            'DataSet:',
            f'wide: {wide_shape}',
            f'tidy: {tidy_shape}',
            f'outputs: {self.outputs}',
            f'inputs: {self.inputs}',
        ])
        return summary

    @property
    def specs(self):
        """Provides keyword arguments for easy instantiation of a similar :class:`DataSet`."""
        return dict(outputs=self.outputs, names_column=self.names_column, values_column=self.values_column,
                    stdzr=self.stdzr, log_vars=self.log_vars, logit_vars=self.logit_vars)

    @property
    def inputs(self):
        """Columns of dataframe not contained in :attr:`outputs`."""
        return [col for col in self.wide.columns if col not in self.outputs]

    @property
    def float_inputs(self):
        """Columns of dataframe with "float64" dtype."""
        return [col for col in self.inputs if self.wide[col].dtype == 'float64']

    @property
    def wide(self) -> WideData:
        """Wide-form view of data"""
        return WideData(self.data, **self.specs)

    @wide.setter
    def wide(self, wide_df: pd.DataFrame):
        assert any([output in wide_df.columns for output in self.outputs]), \
            f'Dataframe must have at least one of outputs {self.outputs}'
        self.data = wide_df

    @property
    def tidy(self) -> TidyData:
        """Tidy-form view of data"""
        return TidyData(self.data, **self.specs)

    @tidy.setter
    def tidy(self, tidy_df: pd.DataFrame):
        assert all([col in tidy_df.columns for col in [self.names_column, self.values_column]]), \
            f'Dataframe must have both columns {[self.names_column, self.values_column]}'
        self.wide = WideData.from_tidy(tidy_df, **self.specs)

    @classmethod
    def from_tidy(cls, tidy, outputs=None, names_column='Variable', values_column='Value',
                  stdzr=None, log_vars=None, logit_vars=None):
        """Constructs a `DataSet` from a tidy-form dataframe. See :class:`DataSet` for explanation of arguments."""
        assert all([col in tidy.columns for col in [names_column, values_column]]), \
            f'Dataframe must have both columns {[names_column, values_column]}'
        specs = dict(outputs=outputs, names_column=names_column, values_column=values_column,
                     stdzr=stdzr, log_vars=log_vars, logit_vars=logit_vars)
        wide = WideData.from_tidy(tidy, **specs)
        return cls(wide, **wide.specs)

    @classmethod
    def from_wide(cls, wide, outputs=None, names_column='Variable', values_column='Value',
                  stdzr=None, log_vars=None, logit_vars=None):
        """Constructs a `DataSet` from a wide-form dataframe. See :class:`DataSet` for explanation of arguments."""
        return cls(wide, outputs=outputs, names_column=names_column, values_column=values_column, stdzr=stdzr,
                   log_vars=log_vars, logit_vars=logit_vars)

    def update_stdzr(self):
        """Updates internal :class:`Standardizer` with current data, :attr:`log_vars`, and :attr:`logit_vars`."""
        self.stdzr |= Standardizer.from_DataFrame(self.wide, log_vars=self.log_vars, logit_vars=self.logit_vars)
