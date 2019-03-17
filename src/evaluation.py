"""
Utilities for evaluating models.
"""

import os
import torch
import numpy as np

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


def generate_episode_data(model, test_loader, vocab):
    """
    Generates the attention map for a single episode.

    Parameters
    ---
    model : torch.Model
        Model used to make the predictions.
    test_loader : torch.DataLoader
        DataLoader to sample a single batch.
    vocab : torchtext.Vocab
        Vocabulary to map back to text.

    Returns
    ---
    bool
        True if the chosen episode was predicted correctly.
    """
    # Get a single batch
    batch = next(iter(test_loader))
    batch_support_set, batch_targets, batch_labels, batch_target_labels = batch

    # Run manually through model...
    support_encodings = model.encode(batch_support_set)
    target_encodings = model.encode(batch_targets)

    # Embed both sets using f() and g()
    batch_support_embeddings = model.g(support_encodings)
    batch_target_embeddings = model.f(target_encodings,
                                      batch_support_embeddings)

    # Compute attention matrix between support and target embeddings
    batch_attention = model._attention(batch_support_embeddings,
                                       batch_target_embeddings)

    # Convert back to text
    logits = model._to_logits(batch_attention, batch_labels)
    predictions = logits.argmax(dim=2)

    # Get an batch element predicted right
    correct_predictions = (predictions == batch_target_labels).view(-1)
    batch_idx = correct_predictions.view(-1).argmax()
    support_set, targets, labels, target_labels = _episode_to_text(
        batch_support_set[batch_idx], batch_targets[batch_idx],
        batch_labels[batch_idx], batch_target_labels[batch_idx], vocab)

    # Save attention map
    attention = batch_attention[batch_idx].detach().numpy()
    support_embeddings = batch_support_embeddings[batch_idx].detach().numpy()
    target_embeddings = batch_target_embeddings[batch_idx].detach().numpy()
    file_name = f"{model.name}_episode.npz"
    file_path = os.path.join(RESULTS_PATH, file_name)
    np.savez(
        file_path,
        attention=attention,
        support_embeddings=support_embeddings,
        target_embeddings=target_embeddings,
        support_set=support_set,
        targets=targets,
        labels=labels,
        target_labels=target_labels)

    return correct_predictions[batch_idx] == 1


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
    flat_support_set = vocab.to_text(flat_support_set)
    support_set = flat_support_set.reshape(N, k)

    targets = vocab.to_text(targets)
    labels = vocab.to_text(labels)
    target_labels = vocab.to_text(target_labels)

    return support_set, targets, labels, target_labels
