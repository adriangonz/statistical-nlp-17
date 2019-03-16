import torch
from torch.nn import functional as F

# Some of these metrics compute to distances
# between elements, thus to convert it to a
# similarity metric we need to take its inverse.
# On cases where the distance may be zero,
# EPS is used instead.
EPS = 1e-8


def distance_to_similarity(distance_func):
    """
    Decorator to transform a distance metric to a similarity
    by taking its reciprocal, or 1/EPS if zero.
    """

    def _inner(*args, **kwargs):
        distance = distance_func(*args, **kwargs) + EPS
        return torch.reciprocal(distance)

    return _inner


@distance_to_similarity
def euclidean_similarity(x, y):
    """
    Computes Euclidean similarity.

    Parameters
    ---
    x, y : torch.Tensor[n x d]
        List of [n] tensors of dimension [d].

    Returns
    ---
    torch.Tensor[n]
        Euclidean similarity for the [n] tensors.
    """
    x = F.normalize(x, dim=1)
    y = F.normalize(y, dim=1)

    return torch.norm(x - y, p=2, dim=1)


@distance_to_similarity
def minkowski_similarity(x, y, p=1.0):
    """
    Computes Minkowski distance between two tensors.

    Parameters
    ---
    x, y : torch.Tensor[n x d]
        List of [n] tensors of dimension [d].
    p : int
        Degree of the p-norm.

    Returns
    ---
    torch.Tensor[n]
        Minkowski similarity for the [n] tensors.
    """
    x = F.normalize(x, dim=1)
    y = F.normalize(y, dim=1)

    return torch.norm(x - y, p=p, dim=1)


def cosine_similarity(x, y):
    """
    Cosine similarity between two tensors

    Parameters
    ---
    x, y : torch.Tensor[n x d]
        List of [n] tensors of dimension [d].

    Returns
    ---
    torch.Tensor[n]
        Cosine similarity for the [n] tensors.
    """
    return F.cosine_similarity(x, y, dim=1)


@distance_to_similarity
def poincare_similarity(x, y):
    """
    Returns Poincare similarity between two tensors.

    Parameters
    ---
    x, y : torch.Tensor[n x d]
        List of [n] tensors of dimension [d].

    Returns
    ---
    torch.Tensor[n]
        Poincare similarity for the [n] tensors.
    """
    x = F.normalize(x, dim=1)
    y = F.normalize(y, dim=1)
    diff_sq_norm = torch.pow(x - y, 2).sum(dim=1)

    alpha = 1 - torch.pow(x, 2).sum(dim=1)
    beta = 1 - torch.pow(y, 2).sum(dim=1)
    inner_term = diff_sq_norm / (alpha * beta)
    return arccosh(1 + 2 * inner_term)


def arccosh(x):
    """
    Element-wise arcosh operation.

    Parameters
    ---
    x : torch.Tensor[]

    Returns
    ---
    torch.Tensor[]
        arcosh result.
    """
    return torch.log(x + torch.sqrt(torch.pow(x, 2) - 1))


SIMILARITY_FUNCTIONS = {
    'cosine': cosine_similarity,
    'euclidean': euclidean_similarity,
    'poincare': poincare_similarity,
    'minkowski': minkowski_similarity
}


def get_similarity_func(similarity_metric):
    """
    Returns the similarity function used.

    Parameters
    ---
    similarity_metric : str
        Similarity metric to use. Needs to be
        one of 'cosine', 'euclidean', 'poincare'
        or 'minkowski'.

    Returns
    ---
    similarity_func : function
        Similarity function.
    """
    return SIMILARITY_FUNCTIONS[similarity_metric]
