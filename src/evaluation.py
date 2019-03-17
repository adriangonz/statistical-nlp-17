"""
Utilities for evaluating models.
"""

import os
import torch
import numpy as np

from .data import reverse_tensor

RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "results")


def predict(model, test_loader):
    """
    Predicts the output of an entire test loader.

    Parameters
    ---
    model : torch.nn.Module
        Pre-trained model.
    test_loader : torch.data.DataLoader
        DataLoader over the test set.

    Returns
    ---
    labels : torch.Tensor[num_episodes]
        Actual labels for each episode.
    predictions : torch.Tensor[num_episodes]
        Predictions for each episode.
    """
    num_episodes = len(test_loader.sampler)
    labels = torch.zeros(num_episodes)
    predictions = torch.zeros(num_episodes)

    prev_idx = 0
    for batch_index, batch in enumerate(test_loader):
        # Make predictions!
        # The first thre parameters are input for the model
        batch_predictions = model(batch[:3])
        # The last one are the correct labels
        target_labels = batch[-1]

        # Get predicted token and flatten
        # NOTE: Here we are assuming a single target per episode!!
        batch_predictions = batch_predictions.argmax(dim=2)
        batch_predictions = batch_predictions.squeeze()
        target_labels = target_labels.squeeze()

        # Store predictions and labels on next index
        batch_size = target_labels.shape[0]
        next_idx = prev_idx + batch_size
        labels[prev_idx:next_idx] = target_labels
        predictions[prev_idx:next_idx] = batch_predictions

        prev_idx = next_idx

    return labels, predictions


def save_predictions(model, labels, predictions):
    """
    Save predictions to a CSV. Note that this method
    works with numpy!

    Parameters
    ---
    model : torch.Model
        Model used to make the predictions. Used for
        generating the file name.
    labels, predictions : np.array[num_episodes]
        Set of labels and predictions.
    """
    file_name = f"{model.name}_predictions.npz"
    file_path = os.path.join(RESULTS_PATH, file_name)

    np.savez(file_path, labels=labels, predictions=predictions)


def generate_attention_map(model, test_loader, vocab):
    """
    Generates the attention map for a single episode.

    Parameters
    ---
    model : torch.Model
        Model used to make the predictions.
    test_loader : torch.data.DataLoader
        DataLoader over the test set.
    vocab : torchtext.Vocab
        Vocabulary to map back to text.
    """
    # Get a single batch
    batch_support_set, batch_targets, batch_labels, batch_target_labels = next(
        iter(test_loader))

    # Run manually through model...
    support_encodings = model.encode(batch_support_set)
    target_encodings = model.encode(batch_targets)

    # Embed both sets using f() and g()
    support_embeddings = model.g(support_encodings)
    target_embeddings = model.f(target_encodings, support_embeddings)

    # Compute attention matrix between support and target embeddings
    batch_attention = model._attention(support_embeddings, target_embeddings)

    # Extract elements of the first episode of the batch
    attention = batch_attention[0].detach().numpy()
    support_set = batch_support_set[0]
    targets = batch_targets[0]
    labels = batch_labels[0]
    target_labels = batch_target_labels[0]

    # Convert back to text
    support_set, targets, labels, target_labels = _episode_to_text(
        batch_support_set[0], batch_targets[0], batch_labels[0],
        batch_target_labels[0], vocab)

    # Save attention map
    file_name = f"{model.name}_attention.npz"
    file_path = os.path.join(RESULTS_PATH, file_name)
    np.savez(
        file_path,
        attention=attention,
        support_set=support_set,
        targets=targets,
        labels=labels,
        target_labels=target_labels)


def _episode_to_text(support_set, targets, labels, target_labels, vocab):
    """
    Converts an episode back to text.

    Parameters
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

    Returns
    ---
    support_set : np.ndarray[N x k]
        Support set formed of the [k] example sentences
        for each of the [N] labels.
    targets : np.ndarray[T]
        Targets to predict for each of the [N] labels.
    labels : torch.Tensor[N]
        List of labels on the episode.
    target_labels : torch.Tensor[num_targets]
        List of right labels of targets.
    """
    # First, we need to flatten these...
    N, k, _ = support_set.shape
    flat_support_set = support_set.view(N * k, -1)
    flat_support_set = reverse_tensor(flat_support_set, vocab)
    support_set = flat_support_set.reshape(N, k)

    targets = reverse_tensor(targets, vocab)
    labels = reverse_tensor(labels, vocab)
    target_labels = reverse_tensor(target_labels, vocab)

    return support_set, targets, labels, target_labels
