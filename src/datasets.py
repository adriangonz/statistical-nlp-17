import torch
import numpy as np

from torch.utils.data.dataset import Dataset
from torch.utils.data import Sampler

from .utils import sample_elements


class EpisodesSampler(Sampler):
    """
    Episodes sampler, which can be used in conjuction with an episodes dataset
    to build a dataloader for meta-training and meta-testing.
    """

    def __init__(self, data_set, N):
        """
        Initialise the episodes sampler.

        Parameters
        ----
        data_set : EpisodesDataset
            Dataset with all the data.
        N : int
            Size of each meta-* episode.
        """
        self.data_set = data_set
        self.N = N

    def __iter__(self):
        """
        Iterates over episodes built out of N labels.

        Returns
        ---
        iterator
            Iterator over groups of indices of labels.
        """
        label_indices = np.arange(len(self.data_set))
        shuffled = np.random.choice(
            label_indices, size=(self.__len__(), self.N))

        return iter(shuffled)

    def __len__(self):
        """
        Return the amount of meta-* episodes that can be built given the size
        of the dataset and the amount of labels we want per dataset.

        Returns
        ---
        int
            Total number of available meta-* episodes.
        """
        # Use floor division
        return len(self.data_set) // self.N


class EpisodesDataset(Dataset):
    """
    Custom implementation of a dataset which is indexed
    by label and returns all the k example sentences.
    """

    def __init__(self, file_path, k):
        """
        Initialise the dataset with the file path where
        all the sentence encodings are stored.

        Parameters
        ----
        file_path : str
            String pointing to the file path, which should contain a dataset
            of shape [num_labels x num_examples x sen_length]
        k : int
            Number of examples per label.
        """
        self.X = np.load(file_path)
        self.k = k

    def __getitem__(self, episode_indices):
        """
        Returns an episode data for a list of labels
        of size N.

        Parameters
        ---
        episode_indices : list
            List of indices of labels of size N to return
            for the episode.

        Returns
        ---
        support_set : torch.Tensor[N x k x sen_length]
            Support set formed of the [k] example sentences of size
            [sen_length] each, for each of the [N] labels.
        targets : torch.Tensor[N x sen_length]
            Targets to predict for each of the N labels.
        labels : torch.Tensor[N]
            List of labels on the episode.
        """
        N = len(episode_indices)
        sen_length = self.X.shape[2]

        support_set = np.zeros((N, self.k, sen_length), dtype=int)
        targets = np.zeros((N, sen_length), dtype=int)
        labels = np.zeros(N, dtype=int)

        for n, idx in enumerate(episode_indices):
            examples, target, label = self._get(idx)

            support_set[n] = examples
            targets[n] = target
            labels[n] = label

        # Transform to torch.Tensor
        tensor_support_set = torch.from_numpy(support_set)
        tensor_targets = torch.from_numpy(targets)
        tensor_labels = torch.from_numpy(labels)

        return tensor_support_set, tensor_targets, tensor_labels

    def _get(self, idx):
        """
        Returns a single label.

        Parameters
        ---
        idx_or_list : int | list[int]
            Index of the label or list of indices.

        Returns
        ---
        support_set : torch.Tensor[k x sen_length x vocab_size]
            [k] example sentences of size [sen_length] each.
        target : torch.Tensor[sen_length x vocab_size]
            Target to predict for the given label.
        label : int
            Index of the label on the dataset.
        """
        sentences = self.X[idx]

        # Sample k sentences for the support set
        # plus an extra one for the target
        sampled_sentences = sample_elements(sentences, size=(self.k + 1))

        support_set = np.array(sampled_sentences[:-1])
        target = np.array(sampled_sentences[-1])

        return support_set, target, idx

    def __len__(self):
        """
        Returns the *length* of the dataset, that is, the number of labels
        available.

        Returns
        ---
        int
            Number of labels.
        """
        return self.X.shape[0]
