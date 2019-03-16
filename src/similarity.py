import torch
from torch.nn import functional as F

# Some of these metrics compute to distances
# between elements, thus to convert it to a
# similarity metric we need to take its inverse.
# On cases where the distance may be zero,
# EPS will be returned instead.
EPS = 1e-8


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
    distance = torch.sqrt(torch.sum(torch.pow(x - y, 2), dim=1)) + EPS
    return torch.reciprocal(distance)


def minkowski_similarity(x, y, p_value=1.0):
    """
    Computes Minkowski distance between two tensors.

    Parameters
    ---
    x, y : torch.Tensor[n x d]
        List of [n] tensors of dimension [d].
    p_value : float
        TODO: What is this?

    Returns
    ---
    torch.Tensor[n]
        Minkowski similarity for the [n] tensors.
    """

    x = F.normalize(x, dim=1)
    y = F.normalize(y, dim=1)
    distance = torch.pow(
        torch.sum(torch.pow(torch.abs(x - y), p_value), dim=1),
        1. / p_value) + EPS

    return torch.reciprocal(distance)


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
    diff = x - y
    diff_norm = (diff * diff).sum(dim=1)
    alpha = 1. - (x * x).sum(dim=1)
    beta = 1. - (y * y).sum(dim=1)
    distance = arccosh(1 + 2 * diff_norm / alpha / beta) + EPS
    return torch.reciprocal(distance)


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
    return torch.log(x + torch.sqrt(torch.add(torch.pow(x, 2), -1.)))


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
