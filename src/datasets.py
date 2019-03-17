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

    def __init__(self, data_set, N, episodes_multiplier=1):
        """
        Initialise the episodes sampler.

        Parameters
        ----
        data_set : EpisodesDataset
            Dataset with all the data.
        N : int
            Size of each meta-* episode.
        episodes_multiplier : int
            Multiply the amount of episodes we iterate over,
            so that variances on samples can be averaged out.
        """
        self.data_set = data_set
        self.N = N
        self.episodes_multiplier = episodes_multiplier

    def __iter__(self):
        """
        Iterates over episodes built out of N labels.

        Returns
        ---
        iterator
            Iterator over groups of indices of labels.
        """
        label_indices = np.arange(len(self.data_set))
        actual_num_episodes = self._get_num_episodes()

        episodes_indices = np.array([
            np.random.choice(
                label_indices, size=(actual_num_episodes, self.N))
            for extra in range(self.episodes_multiplier)
        ]).reshape(-1, self.N)

        return iter(episodes_indices)

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
        num_episodes = self._get_num_episodes()
        return self.episodes_multiplier * num_episodes

    def _get_num_episodes(self):
        """
        Returns the actual number of episodes available (ignoring
        [episodes_multiplier].

        Returns
        ---
        int
            Number of _original_ episodes.
        """
        return len(self.data_set) // self.N


class EpisodesDataset(Dataset):
    """
    Custom implementation of a dataset which is indexed
    by label and returns all the k example sentences.
    """

    def __init__(self, X, y, k, num_targets=1):
        """
        Initialise the dataset with the file path where
        all the sentence encodings are stored.

        Parameters
        ----
        X : torch.Tensor[num_labels x num_examples x sen_length]
            Example sentences from which we will extract both the support
            and target sets.
        y : torch.Tensor[num_labels]
            Value in vocabulary of each of the sentences.
        k : int
            Number of examples per label.
        num_targets : int
            Number of targets per episode.
        shuffle_between_episodes : bool
            Should we shuffle the targets and sentences between
            episodes.
        """
        self.X = X
        self.y = y
        self.k = k
        self.num_targets = num_targets

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
        targets : torch.Tensor[T x sen_length]
            Targets to predict for each of the [N] labels.
        labels : torch.Tensor[N]
            List of labels on the episode.
        target_labels : torch.Tensor[num_targets]
            List of right labels of targets.
        """
        N = len(episode_indices)
        sen_length = self.X.shape[2]

        support_set = torch.zeros((N, self.k, sen_length), dtype=torch.long)
        targets = torch.zeros((N, sen_length), dtype=torch.long)
        labels = torch.zeros(N, dtype=torch.long)

        for n, idx in enumerate(episode_indices):
            examples, target, label = self._get(idx)

            support_set[n] = examples
            targets[n] = target
            labels[n] = label

        # Choose [num_targets] randomly
        target_indices = sample_elements(np.arange(N), size=self.num_targets)

        targets = targets[target_indices]
        target_labels = labels[target_indices]
        return support_set, targets, labels, target_labels

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
            Index of the label on the vocabulary.
        """
        sentences = self.X[idx]

        # Sample k sentences for the support set
        # plus an extra one for the target
        sampled_sentences = sample_elements(sentences, size=(self.k + 1))

        support_set = torch.stack(sampled_sentences[:-1])
        target = sampled_sentences[-1]
        label = self.y[idx]

        return support_set, target, label

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
