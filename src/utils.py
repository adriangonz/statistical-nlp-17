import numpy as np
import torch

from functools import reduce

from sklearn.model_selection import train_test_split

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


def train_test_split_tensors(*tensors, **options):
    """
    Adaptor for torch Tensors to use sklearn's train_test_split method.

    Parameters
    ---
    *tensors : list[torch.Tensor]
        List of torch tensors to split.
    **options : kwargs
        Options to be passed down to sklearn's train_test_split.

    Returns
    ---
    *split : 2 * len(tensors)
        Split tensors.
    """
    indices = [np.arange(tensor.shape[0]) for tensor in tensors]

    split_indices = train_test_split(*indices, **options)

    split_tensors = [
        tensors[n // 2][indices] for n, indices in enumerate(split_indices)
    ]

    return split_tensors


def getattrd(obj, name):
    """
    Same as getattr(), but allows dot notation lookup.

    Discussed in:
    http://stackoverflow.com/questions/11975781
    """
    return reduce(getattr, name.split("."), obj)


def get_model_name(distance='cosine', embeddings='vanilla', N=5, k=3):
    """
    Generates the model name combining its diff parameters.

    Parameters
    ---
    distance : str
        Distance metric used.
    embeddings : str
        Type of word embeddings used.
    N : int
        Number of labels per episode.
    k : int
        Number of examples per label on each episode.
    """
    return f"{distance}_{embeddings}_N={N}_k={k}"


def extract_model_parameters(model_name):
    """
    Extracts the model parameters from a model name generated
    with `get_model_name`.

    Parameters
    ---
    model_name : str
        Name of the model.
    Returns
    ---
    distance : str
        Distance metric used
    embeddings : str
        Type of word embedding used.
    N : int
        Number of labels per episode.
    k : int
        Number of examples per label on each episode.
    """
    model_fields = model_name.split("_")
    distance = model_fields[0]
    embeddings = model_fields[1]
    N = _extract_int(model_fields[2])
    k = _extract_int(model_fields[3])

    return distance, embeddings, N, k


def _extract_int(field):
    """
    Extracts the value of a string formatted as
        [key]=[value]
    (e.g. N=3)

    Parameters
    ---
    field : str
        String formatted as [key]=[value].

    Returns
    ---
    value : int
        Value of the field.
    """
    _, val = field.split("=")
    return int(val)
