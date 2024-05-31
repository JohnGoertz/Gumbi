import numpy as np

from .utils import assert_in
from .arrays import ParameterArray as parray


def make_deltas_parray(*, stdzr, scale, **deltas):
    """Make a parray containing standardized differences for each dimension.

    Primarily used to create bounds for the lengthscale hyperparameters of a GP
    model.
    """
    assert_in("scale", scale, ["transformed", "standardized", "natural"])
    match scale:
        case "transformed":
            deltas = {
                dim: [stdzr.untransform(dim, [v, v * 2]) if v is not None else None for v in vs]
                for dim, vs in deltas.items()
            }
        case "standardized":
            deltas = {
                dim: [stdzr.unstdz(dim, [v, v * 2]) if v is not None else None for v in vs]
                for dim, vs in deltas.items()
            }
        case "natural":
            deltas = {dim: [[v, v * 2] if v is not None else None for v in vs] for dim, vs in deltas.items()}

    deltas = {
        dim: [np.diff(stdzr.stdz(dim, v)) if v is not None else [np.nan] for v in vs] for dim, vs in deltas.items()
    }
    ls_bounds = parray(**deltas, stdzr=stdzr, stdzd=True)
    return ls_bounds