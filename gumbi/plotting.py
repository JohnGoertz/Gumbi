from __future__ import annotations  # Necessary for self-type annotations until Python >3.10
from dataclasses import dataclass
from typing import Callable
import warnings

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.special import logit

from gumbi.aggregation import *
from gumbi.arrays import *
from gumbi.utils import round_to_n, Trigger

__all__ = ['ParrayPlotter']


@dataclass
class ParrayPlotter:
    r"""Wrapper for a ``matplotlib.pyplot`` function; adjusts ticks and labels according to plotter settings.

    Provides a consistent interface to matplotlib plotting functions that allows easy iteration between permutations
    of plotting and tick labeling in  natural, transformed, standardized space. When called on a plotting function,
    a :class:`ParrayPlotter` instance passes pre-formated x and y (and z) arrays to the function as positional
    arguments, along with any additional keyword arguments supplied. :class:`ParrayPlotter` then adjusts tick labels
    according to its *\*_tick_scale* arguments.

    Passing a ``.t`` or ``.z`` child of a parray automatically overrides the respective *_scale* argument. This is
    achieved by inspecting the variable name for a ``'_t'`` or ``'_z'`` suffix, so avoiding using variable names with
    those suffixes to avoid confusion. Note that not all permutations of *\*_scale* and *\*_tick_scale* are
    permitted: *_tick_scale* should generally either match the respective *_scale* argument or be ``'natural'``.

    :class:`ParrayPlotter` also provides a :meth:`colorbar` method that adds a colorbar and reformats its ticks and
    labels according to the *z_scale* and *z_tick_scale* attributes.

    Parameters
    ----------
    x_pa, y_pa: ParameterArray | LayeredArray | np.ndarray
        X and Y arrays. If *z_pa* or *stdzr* are not supplied, x_pa or y_pa must contain a Standardizer instance.
    z_pa: ParameterArray | LayeredArray | np.ndarray, optional
        Z array for 2D plots. If *stdzr* is not supplied, *z_pa*, *x_pa*, or *y_pa* must contain a Standardizer instance.
    stdzr: Standardizer, optional
        Standardizer for converting ticks. Only optional if *z_pa*, *x_pa*, or *y_pa* contain a Standardizer instance.
    x_scale, y_scale, z_scale : {'natural', 'transformed', 'standardized'}
        Space in which to plot respective array. Ignored if array is not a :class:`ParameterArray`.
    x_tick_scale, y_tick_scale, z_tick_scale : {'natural', 'transformed', 'standardized'}
        Space in which to label ticks for respective axis. Should be 'natural' or match respective *\*_scale* argument.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from gumbi import Standardizer, ParrayPlotter, ParameterArray
    >>> stdzr = Standardizer(x = {'μ': -5, 'σ': 0.5},
    ...                      y = {'μ': -0.3, 'σ': 0.15},
    ...                      z={'μ': 2, 'σ': 2},
    ...                      log_vars=['x', 'y'], logit_vars=['z'])
    >>> x = np.arange(1, 10, 0.25)
    >>> y = np.arange(1, 10, 0.25)
    >>> x, y = np.meshgrid(x, y)
    >>> z = np.sin(np.sqrt((x - 5) ** 2 + (y - 5) ** 2))**2*0.9+0.05
    >>> xyz = ParameterArray(x=x, y=y, z=z, stdzr=stdzr)

    Make a natural-space contour plot with user-specified levels

    >>> pp = ParrayPlotter(xyz['x'], xyz['y'], xyz['z'])
    >>> pp(plt.contour, levels=8)

    Use the same :class:`ParrayPlotter` to make a different pseudocolor plot and add a colorbar:

    >>> pcm = pp(plt.pcolormesh, shading='gouraud')
    >>> cbar = pp.colorbar(pcm, ax=plt.gca())

    Make a filled contour plot with *x* plotted in natural-space and *x* tick labels displayed in natural-space,
    *y* plotted in transformed space but *y* tick lables displayed in natural-space, and *z* plotted in standardized
    space with a colorbar displaying standardized-space tick labels:

    >>> pp = ParrayPlotter(xyz['x'], xyz['y'].t, xyz['z'], z_scale='standardized', z_tick_scale='standardized')
    >>> cs = pp(plt.contourf)
    >>> cbar = pp.colorbar(cs)
    """
    x: ParameterArray | LayeredArray | np.ndarray
    y: UncertainParameterArray | UncertainArray | ParameterArray | LayeredArray | np.ndarray
    z: UncertainParameterArray | UncertainArray | ParameterArray | LayeredArray | np.ndarray = None
    stdzr: Standardizer = None
    x_scale: str = 'natural'
    x_tick_scale: str = 'natural'
    y_scale: str = 'natural'
    y_tick_scale: str = 'natural'
    z_scale: str = 'natural'
    z_tick_scale: str = 'natural'

    def __post_init__(self):
        self.update()

        for arr in [self.z, self.y, self.x]:
            if self.stdzr is None:
                self.stdzr = getattr(arr, 'stdzr', None)

        if self.stdzr is None:
            raise ValueError('Standardizer must be provided if none of the arrays contain a Standardizer.')

    def update(self):
        self._update_x()
        self._update_y()
        if self.z is not None:
            self._update_z()
        else:
            self.zlabel = None
            self.z_ = None

    def _update_x(self):
        self.x_, self.xlabel, self.x_scale = _parse_array(self.x, self.x_scale)

    def _update_y(self):
        self.y_, self.ylabel, self.y_scale = _parse_array(self.y, self.y_scale)

    def _update_z(self):
        self.z_, self.zlabel, self.z_scale = _parse_array(self.z, self.z_scale)

    def __call__(self, plotter: Callable, **kwargs):
        r"""Wrapper for a ``matplotlib.pyplot`` function; adjusts ticks and labels according to plotter settings.

        Parameters
        ----------
        plotter: Callable
            Plotting function to be wrapped. Must accept at least two or three positional arguments.
        **kwargs
            Additional keyword arguments passed to wrapped function.

        Returns
        -------
        output
            Output of wrapped function
        """
        args = [arg for arg in [self.x_, self.y_, self.z_] if arg is not None]
        out = plotter(*args, **kwargs)
        ax = kwargs.get('ax', plt.gca())
        _format_parray_plot_labels(ax, self.stdzr, self.xlabel, self.x_scale, self.x_tick_scale, self.ylabel,
                                   self.y_scale, self.y_tick_scale)
        return out

    def colorbar(self, mappable=None, cax=None, ax=None, **kwargs):
        """Wrapper for ``matplotlib.pyplot.colorbar``; adjusts ticks and labels according to plotter settings."""
        cbar = plt.colorbar(mappable=mappable, cax=cax, ax=ax, **kwargs)

        self.zlabel = self.zlabel.removesuffix('_z').removesuffix('_t')
        _reformat_tick_labels(cbar, 'c', self.zlabel, self.z_scale, self.z_tick_scale, self.stdzr)

        label = _augment_label(self.stdzr, self.zlabel, self.z_tick_scale)
        cbar.set_label(label)

        return cbar

    def plot(self, ci=0.95, ax=None, palette=None, line_kws=None, ci_kws=None):
        r"""

        Parameters
        ----------
        ci : float or None, default 0.95
            Confidence interval on :math:`0<\mathtt{ci}<1`. If None, no confidence intervals will be drawn.
        ax : plt.Axes, optional
            Axes on which to plot. Defaults to ``plt.gca()``.
        palette : str or array-like
            Name of seaborn palette or list of colors (at least two) for plotting.
        line_kws : dict, optional
            Additional keyword arguments passed to ``plt.plot``.
        ci_kws : dict, optional
            Additional keyword arguments passed to :meth:``plot_ci``.

        Returns
        -------
        ax : plt.Axes
            Axes for the plot
        """
        if self.z is not None:
            raise NotImplementedError('Method "plot" not implemented when z_pa is present.')

        line_kws = dict() if line_kws is None else line_kws
        ci_kws = dict() if ci_kws is None else ci_kws
        palette = sns.cubehelix_palette() if palette is None else palette
        palette = sns.color_palette(palette) if type(palette) is str else palette

        line_defaults = dict(lw=2, color=palette[-2], zorder=0)
        ci_defaults = dict(lw=2, facecolor=palette[1], zorder=-1, alpha=0.5)

        line_kws = line_defaults | line_kws
        ci_kws = ci_defaults | ci_kws

        ax = plt.gca() if ax is None else ax
        ax.plot(self.x_, self.y_, **line_kws)
        if ci is not None and hasattr(self.y, 'σ2'):
            self.plot_ci(ci=ci, ax=ax, **ci_kws)

        _format_parray_plot_labels(ax, self.stdzr, self.xlabel, self.x_scale, self.x_tick_scale, self.ylabel,
                                   self.y_scale, self.y_tick_scale)

        return ax

    def plot_ci(self, ci=0.95, ci_style='fill', center='median', ax=None, **kwargs):
        r"""Plots the confidence interval for an UncertainParameterArray.

        Parameters
        ----------
        ci : float or None, default 0.95
            Confidence interval on :math:`0<\mathtt{ci}<1`. If None, no confidence intervals will be drawn.
        ci_style : {'fill', 'band', 'errorbar', 'bar'}
            Whether to plot CI using ``plt.fill_between`` (*fill* or *band*) or ``plt.errorbar`` (*errorbar* or *bar*).
        center : {'median', 'mean'}
            Which metric to plot as midpoint if using ``plt.errorbar``.
        ax : plt.Axes, optional
            Axes on which to plot. Defaults to ``plt.gca()``.
        **kwargs
            Additional keyword arguments passed to ``plt.fill_between`` or ``plt.errorbar``.

        Returns
        -------
        ax : plt.Axes
            Axes for the plot
        """
        if self.z is not None:
            raise NotImplementedError('Method "plot_ci" not supported when z_pa is present.')
        if not hasattr(self.y, 'σ2'):
            raise NotImplementedError('Method "plot_ci" only supported when y_pa has the "σ2" attribute.')

        ax = plt.gca() if ax is None else ax

        y, *_ = _parse_uparray(self.y, self.y_scale)

        l = y.dist.ppf((1 - ci) / 2)
        m = y.dist.ppf(0.5) if center == 'median' else y.μ
        u = y.dist.ppf((1 + ci) / 2)

        fill_between_styles = ['fill', 'band']
        errorbar_styles = ['errorbar', 'bar']
        if ci_style in fill_between_styles:
            ax.fill_between(self.x_, l, u, **kwargs)
        elif ci_style in errorbar_styles:
            ax.errorbar(self.x_, m, m-l, u-m, **kwargs)
        else:
            return ValueError(f'ci_style must be one of {fill_between_styles + errorbar_styles}')
        return ax


def _parse_array(array, scale) -> (np.ndarray, str, str):
    if isinstance(array, (UncertainParameterArray, UncertainArray)):
        array, label, scale = _parse_uparray(array, scale)
        array = array.μ
    elif isinstance(array, (ParameterArray, LayeredArray)):
        array, label, scale = _parse_parray(array, scale)
        array = array.values()
    else:
        array, label, scale = _parse_parray(array, scale)
    return array, label, scale


def _parse_parray(pa, scale) -> (ParameterArray | LayeredArray | np.ndarray, str, str):
    if isinstance(pa, ParameterArray):
        if scale == 'standardized':
            array = pa.z
        elif scale == 'transformed':
            array = pa.t
        else:
            array = pa
        label = pa.names[0]
    elif isinstance(pa, LayeredArray):
        array = pa
        label = pa.names[0]
        if pa.names[0].endswith('_z'):
            scale = 'standardized'
        elif pa.names[0].endswith('_t'):
            scale = 'transformed'
    else:
        array = pa
        label = ''
    return array, label, scale


def _parse_uparray(upa, scale) -> (UncertainParameterArray | UncertainArray, str, str):
    if isinstance(upa, UncertainParameterArray):
        if scale == 'standardized':
            array = upa.z
        elif scale == 'transformed':
            array = upa.t
        else:
            array = upa
    elif isinstance(upa, UncertainArray):
        if upa.name.endswith('_z'):
            scale = 'standardized'
        elif upa.name.endswith('_t'):
            scale = 'transformed'
        array = upa
    else:
        raise TypeError('Array must be either an UncertainParameterArray or an UncertainArray.')
    label = upa.name

    return array, label, scale


def _format_parray_plot_labels(ax, stdzr, xlabel, x_scale, x_tick_scale, ylabel, y_scale, y_tick_scale):
    xlabel = xlabel.removesuffix('_z').removesuffix('_t')
    ylabel = ylabel.removesuffix('_z').removesuffix('_t')
    _reformat_tick_labels(ax, 'x', xlabel, x_scale, x_tick_scale, stdzr)
    _reformat_tick_labels(ax, 'y', ylabel, y_scale, y_tick_scale, stdzr)

    label = _augment_label(stdzr, xlabel, x_tick_scale)
    ax.set_xlabel(label)

    label = _augment_label(stdzr, ylabel, y_tick_scale)
    ax.set_ylabel(label)

def _augment_label(stdzr, label, tick_scale):
    prefixes = {np.log: 'log ', logit: 'logit '}
    transform = stdzr.transforms.get(label, [None])[0]
    prefix = prefixes.get(transform, '') if tick_scale in ['transformed', 'standardized'] else ''
    suffix = ' (standardized)' if tick_scale == 'standardized' else ''
    return f'{prefix}{label}{suffix}'

def _reformat_tick_labels(ax, axis, name, current, new, stdzr, sigfigs=3):
    tick_setters = {
        # ('natural', 'standardized'): _n_ticks_z_labels,
        # ('natural', 'transformed'): _n_ticks_t_labels,
        ('standardized', 'natural'): _z_ticks_n_labels,
        ('transformed', 'natural'): _t_ticks_n_labels,
    }

    if current != new:
        if (tpl := (current, new)) not in tick_setters:
            raise ValueError('Cannot convert ticks between {0} and {1}'.format(*tpl))
        else:
            tick_setter = tick_setters[tpl]
            tick_setter(ax, axis, stdzr, name, sigfigs=sigfigs)


def _get_ticks_setter(ax, axis):
    if axis == 'x':
        ticks = ax.get_xticks()
        set_ticks = ax.set_xticks
        set_labels = ax.set_xticklabels
    elif axis == 'y':
        ticks = ax.get_yticks()
        set_ticks = ax.set_yticks
        set_labels = ax.set_yticklabels
    elif axis == 'z':
        ticks = ax.get_zticks()
        set_ticks = ax.set_zticks
        set_labels = ax.set_zticklabels
    elif axis == 'c':
        ticks = ax.get_ticks()
        set_ticks = ax.set_ticks
        set_labels = ax.set_ticklabels

    def setter(*args, **kwargs):
        # TODO: Find a better way to set tick labels
        # Setting only labels throws a FixedLocator warning, but setting ticks first extends the plot area excessively
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            set_labels(*args, **kwargs)
        # set_ticks(ticks)
        # set_labels(*args, **kwargs)

    return ticks, setter


def _get_label_setter(ax, axis):
    if axis == 'x':
        set_label = ax.set_xlabel
    elif axis == 'y':
        set_label = ax.set_ylabel
    elif axis == 'z':
        set_label = ax.set_zlabel
    elif axis == 'c':
        set_label = ax.set_label
    return set_label


def _n_ticks_z_labels(ax, axis, stdzr, name, sigfigs=3):
    ticks, set_ticklabels = _get_ticks_setter(ax, axis)
    new_ticks = stdzr.stdz(name, ticks)
    new_ticks = round_to_n(new_ticks, sigfigs)
    set_ticklabels(new_ticks)


def _n_ticks_t_labels(ax, axis, stdzr, name, sigfigs=3):
    ticks, set_ticklabels = _get_ticks_setter(ax, axis)
    new_ticks = stdzr.transform(name, ticks)
    new_ticks = round_to_n(new_ticks, sigfigs)
    set_ticklabels(new_ticks)


def _z_ticks_n_labels(ax, axis, stdzr, name, sigfigs=3):
    ticks, set_ticklabels = _get_ticks_setter(ax, axis)
    new_ticks = stdzr.unstdz(name, ticks)
    new_ticks = round_to_n(new_ticks, sigfigs)
    set_ticklabels(new_ticks)


def _t_ticks_n_labels(ax, axis, stdzr, name, sigfigs=3):
    ticks, set_ticklabels = _get_ticks_setter(ax, axis)
    new_ticks = stdzr.untransform(name, ticks)
    new_ticks = round_to_n(new_ticks, sigfigs)
    set_ticklabels(new_ticks)
