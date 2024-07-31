from scipy.spatial.distance import pdist
import numpy as np
import pymc as pm
from .misc import listify, first
from warnings import warn


def get_ls_prior(X, *, ARD, lower=None, upper=None, mass=0.98):

    if ARD:
        all_points = np.hsplit(X, X.shape[1])
    else:
        all_points = [X]

    lowers = [None] if lower is None else listify(lower)
    if len(lowers) == 1:
        lowers = lowers * len(all_points)
    if len(lowers) != len(all_points):
        raise ValueError("Number of lower bounds must match number of dimensions")

    uppers = [None] if upper is None else listify(upper)
    if len(uppers) == 1:
        uppers = uppers * len(all_points)
    if len(uppers) != len(all_points):
        raise ValueError("Number of upper bounds must match number of dimensions")

    params = []

    for i, (points, lower_, upper_) in enumerate(zip(all_points, lowers, uppers)):
        distances = pdist(points)
        distinct = distances != 0
        
        default_lower = distances[distinct].min() if sum(distinct) > 0 else 0.1

        if lower_ is None:
            lower_ = default_lower
        lower_ = max(lower_, default_lower, 0.01)
        if upper_ is None:
            upper_ = distances[distinct].max() if sum(distinct) > 0 else 1

        converged = False
        mass_ = mass
        while not converged:
            try:
                params_ls = pm.find_constrained_prior(
                    distribution=pm.InverseGamma,
                    lower=lower_,
                    upper=upper_,
                    init_guess={"alpha": lower_, "beta": upper_},
                    mass=mass_,
                )
            except ValueError as e:
                if 'Optimization of parameters failed' in str(e):
                    mass_ -= 0.01
                else:
                    raise e
            else:
                converged = True
                if mass_ != mass:
                    warn("Mass of constrained lengthscale prior was "
                         f"reduced from {mass:.3f} to {mass_:.3f} to enable "
                         f"convergence for dimension {i}.")

        params.append(params_ls)

    params = {k: [dim[k] for dim in params] for k in first(params).keys()}

    return params


# def get_ls_prior(points):
#     distances = pdist(points[:, None])
#     distinct = distances != 0
#     ls_l = distances[distinct].min() if sum(distinct) > 0 else 0.1
#     ls_u = distances[distinct].max() if sum(distinct) > 0 else 1
#     ls_σ = max(0.1, (ls_u - ls_l) / 6)
#     ls_μ = ls_l + 3 * ls_σ
#     return ls_μ, ls_σ
