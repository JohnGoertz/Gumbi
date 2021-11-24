"""Generic utility functions"""

import os
import pathlib as pl
import matplotlib.pyplot as plt


def setup_paths(make_missing=True):
    """Gathers paths of expected directories in experiment structure

    Parameters
    ----------
    make_missing: bool, default True
        Whether to create directories that are expected but not found

    Returns
    -------
    dict
        Pathlib objects for each directory: 'Base' (root directory for experiment), 'Code', 'Data', 'Results', and
        'Figures'
    """
    code_pth = pl.Path(os.getcwd())
    base_pth = code_pth.parent
    data_pth = base_pth / 'Data'
    rslt_pth = base_pth / 'Results'
    fig_pth = base_pth / 'Figures'
    if make_missing:
        data_pth.mkdir(parents=True, exist_ok=True)
        rslt_pth.mkdir(parents=True, exist_ok=True)
        fig_pth.mkdir(parents=True, exist_ok=True)
    return base_pth, code_pth, data_pth, rslt_pth, fig_pth


def savefig(filename: str, fig=None, path=None, silent=False, **kwargs):
    """Saves figure as .png (300 dpi, transparent) and .svg

    Parameters
    ----------
    filename : str
        Name of file to save (without extension)
    fig : mpl.figures.Figure, optional
        Figure object to save. If `None`, calls `plt.gcf()`
    path : pl.Path, optional
        Directory in which to save figure
    silent : bool, default False
        Whether to display figure saving progress
    **kwargs
        Additional arguments passed to `mpl.figures.Figure.savefig()`
    """

    fig = plt.gcf() if fig is None else fig
    path = setup_paths(make_missing=False)[-1] if path is None else path

    kwargs.setdefault('bbox_inches', 'tight')
    kwargs.setdefault('transparent', True)

    if not silent:
        print('Saving.', end='')
    fig.savefig(path / (filename + '.png'), dpi=300, **kwargs)
    if not silent:
        print('.', end='')
    fig.savefig(path / (filename + '.svg'), **kwargs)
    if not silent:
        print('Done')
