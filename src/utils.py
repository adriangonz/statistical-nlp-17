import numpy as np
from functools import reduce


def sample_elements(elements, size):
    """
    Samples a certain number of elements
    out of a regular array.

    Parameters
    ---
    elements : list
        List with elements of any sort.
    size : int
        How many elements to sample.

    Returns
    ---
    list
        List with all sampled elements.
    """
    elements_size = len(elements)
    if size > elements_size:
        raise ValueError(
            f"Can't sample {size} elements out of {elements_size}")

    all_indices = np.arange(elements_size)
    sampled_indices = np.random.choice(all_indices, replace=False, size=size)

    return [elements[idx] for idx in sampled_indices]


def getattrd(obj, name):
    """
    Same as getattr(), but allows dot notation lookup.

    Discussed in:
    http://stackoverflow.com/questions/11975781
    """
    return reduce(getattr, name.split("."), obj)
