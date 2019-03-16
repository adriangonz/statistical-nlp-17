"""
Utilities for evaluating models.
"""

import torch


def predict(model, test_loader, runs=20):
    """
    Predicts the output of an entire test loader.

    Parameters
    ---
    model : torch.nn.Module
        Pre-trained model.
    test_loader : torch.data.DataLoader
        DataLoader over the test set.
    runs : int
        Given that the episodes are created
        stochastically, we need to test it
        through diff combinations.

    Returns
    ---
    labels : torch.Tensor[num_runs * num_episodes]
        Actual labels for each episode.
    predictions : torch.Tensor[num_runs * num_episodes]
        Predictions for each episode.
    """
    num_episodes = len(test_loader.sampler)
    labels = torch.zeros(runs * num_episodes)
    predictions = torch.zeros(runs * num_episodes)

    prev_idx = 0
    for run in range(runs):
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
