import numpy as np
import torch

from functools import reduce
from torch.autograd import Variable


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


def to_one_hot(y, depth=None):
    """
    Transform Tensor into one-hot vector.

    From https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/24

    Parameters
    ---
    y : torch.Tensor
    depth : int
        Length of one-hot vector.

    Returns
    ---
    one_hot : torch.Tensor[]
    """
    y_tensor = y.data if isinstance(y, Variable) else y
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    depth = depth if depth is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], depth).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*(tuple(y.shape) + (-1, )))
    return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot


def getattrd(obj, name):
    """
    Same as getattr(), but allows dot notation lookup.

    Discussed in:
    http://stackoverflow.com/questions/11975781
    """
    return reduce(getattr, name.split("."), obj)
