import torch

from ..similarity import (euclidean_similarity, minkowski_similarity,
                          cosine_similarity, poincare_similarity)


def test_euclidean_similarity(x, y):
    euclidean = euclidean_similarity(x, y)
    expected = torch.Tensor([-1.6847, -0])

    assert torch.allclose(euclidean, expected, rtol=1e-4)


def test_minkowski_similarity(x, y):
    minkowski = minkowski_similarity(x, y)
    expected = torch.Tensor([-1.8528, -0])

    assert torch.allclose(minkowski, expected, rtol=1e-4)


def test_cosine_similarity(x, y):
    cosine = cosine_similarity(x, y)
    expected = torch.Tensor([-0.4191, 1])

    assert torch.allclose(cosine, expected, rtol=1e-4)


def test_poincare_similarity(x, y):
    poincare = poincare_similarity(x, y)
    expected = torch.Tensor([-25.4526, 0.0000])

    assert torch.allclose(poincare, expected)
