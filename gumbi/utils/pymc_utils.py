"""Utility functions for working with pymc3"""

import pymc3 as pm


def nc_Normal(name: str, mu: float, sigma: float, **kws):
    """Constructs non-centered normal distribution via :math:`\mathcal{N}(0,1)\cdot\sigma+\mu`

    Parameters
    ----------
    name: str
        Name given to pymc3 RV.
    mu: float
        Mean.
    sigma: float
        Standard deviation
    **kws:
        Additional arguments passed to pm.Normal.

    Returns
    -------
    rv: pm.Deterministic
        Non-centered Normal distribution
    nc: pm.Normal
        Underlying standard Normal distribution, named with '_nc' appended

    """
    nc = pm.Normal(name+'_nc', mu=0, sigma=1, **kws)
    rv = pm.Deterministic(name, mu + sigma*nc)

    return rv, nc


def sc_Exponential(name: str, mu: float, **kws):
    """Scaled Exponential distribution: :math:`\text{Exponential}(1)\cdot\mu

    Parameters
    ----------
    name: str
        Name given to pymc3 RV.
    mu: float
        Mean.
    **kws:
        Additional arguments passed to pm.Exponential.

    Returns
    -------
    rv: pm.Deterministic
        Scaled Exponential distribution
    nc: pm.Exponential
        Underlying standard Exponential distribution, named with '_nc' appended

    """
    nc = pm.Exponential(name+'_nc', lam=1, **kws)
    rv = pm.Deterministic(name, mu*nc)
    return rv, nc
