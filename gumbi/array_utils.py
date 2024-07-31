import numpy as np

from .utils import assert_in, first, one
from .arrays import ParameterArray as parray
from .arrays import UncertainParameterArray as uparray


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


def stack(array_list, axis=0, **kwargs):
    ndims = {pa.ndim for pa in array_list}
    if ndims == {1}:
        return hstack(array_list)
    types = {type(pa) for pa in array_list}
    if not len(types) == 1:
        raise ValueError("Arrays are not all of the same type.")
    cls = one(types)
    if cls == parray:
        return _stack_parray(array_list, **kwargs)
    elif cls == uparray:
        all_names = [upa.name for upa in array_list]
        if not len(set(all_names)) == 1:
            raise ValueError("Arrays do not have the same name.")
    else:
        raise ValueError(f"Unknown array type: {cls}")
    new = np.stack(array_list, axis=axis, **kwargs)
    stdzr = _get_stdzr(array_list)
    return cls(**{dim: new[dim] for dim in new.dtype.names}, stdzr=stdzr)


def vstack(array_list, **kwargs):
    types = {type(pa) for pa in array_list}
    if not len(types) == 1:
        raise ValueError("Arrays are not all of the same type.")
    cls = one(types)
    if cls == parray:
        return _vstack_parray(array_list, **kwargs)
    elif cls == uparray:
        all_names = [upa.name for upa in array_list]
        if not len(set(all_names)) == 1:
            raise ValueError("Arrays do not have the same name.")
    else:
        raise ValueError(f"Unknown array type: {cls}")
    new = np.vstack(array_list, **kwargs)
    stdzr = _get_stdzr(array_list)
    return cls(**{dim: new[dim] for dim in new.dtype.names}, stdzr=stdzr)


def hstack(array_list, **kwargs):
    types = {type(pa) for pa in array_list}
    if not len(types) == 1:
        raise ValueError("Arrays are not all of the same type.")
    cls = one(types)
    if cls == parray:
        return _hstack_parray(array_list, **kwargs)
    elif cls == uparray:
        all_names = [upa.name for upa in array_list]
        if not len(set(all_names)) == 1:
            raise ValueError("Arrays do not have the same name.")
    else:
        raise ValueError(f"Unknown array type: {cls}")
    new = np.hstack(array_list, **kwargs)
    stdzr = _get_stdzr(array_list)
    return cls(**{dim: new[dim] for dim in new.dtype.names}, stdzr=stdzr)


def _stack_parray(array_list, axis=0, **kwargs):
    _check_parrays_compatible(array_list)
    new = np.stack(array_list, axis=axis, **kwargs)
    stdzr = _get_stdzr(array_list)
    return parray(**{dim: new[dim] for dim in new.dtype.names}, stdzr=stdzr)


def _vstack_parray(array_list, **kwargs):
    _check_parrays_compatible(array_list)
    new = np.vstack(array_list, **kwargs)
    stdzr = _get_stdzr(array_list)
    return parray(**{dim: new[dim] for dim in new.dtype.names}, stdzr=stdzr)


def _hstack_parray(array_list, **kwargs):
    _check_parrays_compatible(array_list)
    new = np.hstack(array_list, **kwargs)
    stdzr = _get_stdzr(array_list)
    return parray(**{dim: new[dim] for dim in new.dtype.names}, stdzr=stdzr)


def _check_parrays_compatible(array_list):
    all_names = [tuple(pa.names) for pa in array_list]
    if not len(set(all_names)) == 1:
        raise ValueError("Arrays do not have the same names.")
    if not all(names == first(all_names) for names in all_names):
        raise ValueError("Arrays names are not in the same order.")


def _get_stdzr(array_list):
    stdzr = first(array_list).stdzr
    if not all(a.stdzr is stdzr for a in array_list):
        raise ValueError("Arrays do not have the same standardizer.")
    return stdzr
