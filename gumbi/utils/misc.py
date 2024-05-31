"""Internal utility functions, not intended for public consumption"""

from functools import wraps
from itertools import chain, islice
from typing import Iterable, Iterator
from operator import attrgetter

import numpy as np


class NotExactlyOneError(Exception):
    """Raised when more than one value is found in a list."""

    pass


def one(itr: Iterable):
    """Ensures list has exactly one element, then returns it."""
    if isinstance(itr, (set, Iterator)):
        itr = list(itr)
    if (n := len(itr)) != 1:
        raise NotExactlyOneError(f"Expected one element in list, got {n}")
    return listify(itr)[0]


def first(itr: Iterable):
    """Returns first element in list."""
    return listify(itr)[0]


def extract(attr, itr):
    """Extracts attribute from each element in iterable."""
    return list(map(attrgetter(attr), itr))


def listify(x):
    """Wraps input in a list if it's not already one, or converts an iterable."""
    if isinstance(x, list):
        return x
    elif isinstance(x, str):
        return [x]
    elif isinstance(x, (set, Iterator, Iterable)):
        return list(x)
    elif x is None:
        return []
    else:
        return [x]


def flatten(list_of_lists, depth=-1):
    """Flatten specified number of levels of nesting. If depth is -1, flatten all."""
    match depth:
        case 0:
            return list_of_lists
        case -1:
            if not isinstance(first(list_of_lists), list):
                return list_of_lists
            else:
                depth = 0
        case _:
            pass
    flatter = list(chain.from_iterable(list_of_lists))
    return flatten(flatter, depth - 1)


def group_by(itr, key, unique=False):
    """Groups elements in iterable by key."""
    if isinstance(key, str):
        key = attrgetter(key)
    if unique:
        one_ = one
    else:
        one_ = skip
    groups = {group: one_([el for el in itr if key(el) == group]) for group in set(map(key, itr))}
    return groups


def skip(x):
    """Returns input unmodified."""
    return x


def NotImplementedWrapper(func):
    """Blocks execution and raises an error."""

    @wraps(func)
    def block(*args, **kwargs):
        """Raises error"""
        raise NotImplementedError

    return block


def assert_in(name: str, arg, itr: Iterable):
    """Raises error if value not in list."""
    if arg not in itr:
        raise ValueError(f"{name} must be one of {itr}")


def assert_is_subset(name: str, subset: Iterable, superset: Iterable):
    """Raises error if any required value not in list."""
    l_set = set(superset)
    r_set = set(subset)
    if not l_set.issuperset(r_set):
        missing = list(r_set.difference(l_set))
        msg = f"{list_is_are(missing)} missing from {name}"
        raise ValueError(msg)


def assert_one(names: str, itr: Iterable):
    """Raises error if none or multiple of values in `itr` are not None."""
    if sum(el is not None for el in itr) != 1:
        raise ValueError(f"Exactly one of {names} must be supplied")


def list_is_are(lst: list) -> str:
    """String formats list to be grammatically correct."""
    lst = listify(lst)
    if len(lst) == 0:
        return None
    elif len(lst) == 1:
        msg = f"{list_and(lst)} is"
    else:
        msg = f"{list_and(lst)} are"
    return msg


def list_and(lst: list) -> str:
    """String formats list to insert 'and' before last element."""
    lst = listify(lst)
    if len(lst) == 0:
        msg = ""
    elif len(lst) == 1:
        msg = f"{lst[0]}"
    elif len(lst) == 2:
        msg = f"{lst[0]} and {lst[1]}"
    else:
        msg = f'{", ".join(lst[:-1])}, and {lst[-1]}'
    return msg


def s(n):
    return "s" if n != 1 else ""


def round_to_n(x, n=2):
    """Round to a specified number of significant digits."""
    if isinstance(x, float):
        rounded = np.round(x, -int(np.floor(np.log10(np.abs(x))) - (n - 1)))
    elif isinstance(x, (list, np.ndarray)):
        rounded = [(np.round(x_, -int(np.floor(np.log10(np.abs(x_))) - (n - 1))) if x_ != 0 else 0) for x_ in x]
    else:
        raise ValueError("x must be float, list, or ndarray.")
    return np.where(x == 0.0, x, rounded)


def prettyprint_dict(dct, lpad=2):
    # Determine left-padding by length of longest key
    m = max(map(len, list(dct.keys()))) + lpad
    lines = []
    for k, v in dct.items():
        left = k.rjust(m)
        if isinstance(v, str):
            right = v
        else:
            right = np.array2string(np.array(v), prefix=(k.rjust(m) + ": "))
        line = f"{left}: {right}"
        lines.append(line)
    return "\n".join(lines)


def batched(iterable, n):
    """Yield successive n-sized batches from iterable."""
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


class Trigger(object):
    """Descriptor to create a property that calls an instance method when its value is changed.
    Note, must be specified as a class variable.

    Parameters
    ----------
    method: str
        The name of the instance method to be called
    default: optional
        Default value of variable
    """

    def __init__(self, method, default=None):
        self.default = default
        self.method = method
        self.data = dict()

    def __get__(self, instance, owner):
        return self.data.get(instance, self.default)

    def __set__(self, instance, value):
        self.data[instance] = value
        getattr(instance, self.method)()


class InstanceCopy:
    """A class that creates instances by copying all attributes of a parent instance"""

    def __init__(self, parent):
        assert isinstance(parent, self.__class__.__bases__[-1])

        for attr in parent.__dict__:
            setattr(self, attr, getattr(parent, attr))
