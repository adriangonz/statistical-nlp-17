import torch

from ..similarity import (euclidean_similarity, minkowski_similarity,
                          cosine_similarity, poincare_similarity, EPS)


def test_euclidean_similarity(x, y):
    euclidean = euclidean_similarity(x, y)
    expected = torch.Tensor([0.59359, 1 / EPS])

    assert torch.allclose(euclidean, expected)


def test_minkowski_similarity(x, y):
    minkowski = minkowski_similarity(x, y)
    expected = torch.Tensor([0.53973, 1 / EPS])

    assert torch.allclose(minkowski, expected)


def test_cosine_similarity(x, y):
    cosine = cosine_similarity(x, y)
    expected = torch.Tensor([-0.4191, 1])

    assert torch.allclose(cosine, expected, rtol=1e-4)


def test_poincare_similarity(x, y):
    poincare = poincare_similarity(x, y)
    expected = torch.Tensor([0, 100000000])

    assert torch.allclose(poincare, expected)
