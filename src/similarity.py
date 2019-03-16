import torch
from torch import nn
from torch.nn import functional as F


class Similarity():
    """ Five similarity measures function """

    def __init__(self):
        self.eps=1e-8

    def euclidean_similarity(self, x, y):
        """ return euclidean distance between two tensors"""
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        distance = torch.sqrt(torch.sum(torch.pow(x-y,2), dim=1)) + self.eps
        return torch.reciprocal(distance)


    def minkowski_similarity(self, x, y, p_value):
        """ return minkowski distance between two tensors"""

        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        distance = torch.pow(torch.sum(torch.pow(torch.abs(x - y), p_value), dim=1),1./p_value) + self.eps
        
        return torch.reciprocal(distance)


    def cosine_similarity(self, x, y):
        """ return cosine similarity between two tensors"""

        return F.cosine_similarity(x, y, dim=1)



    def poincare_similarity(self, x, y):
        """ return poincare distance between two tensors"""
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        diff = x-y
        diff_norm = (diff*diff).sum(dim=1)
        alpha = 1. - (x*x).sum(dim=1)
        beta  = 1. - (y*y).sum(dim=1)
        distance = self.arccosh(1 + 2 * diff_norm / alpha / beta) + self.eps
        return torch.reciprocal(distance)


    def arccosh(self, x):
        """elementwise arcosh operation.
        """
        return torch.log(x + torch.sqrt(torch.add(torch.pow(x, 2), -1.)))