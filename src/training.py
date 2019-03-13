from torch import optim
from torch.nn import functional as F

from ignite.engine import (Events, create_supervised_trainer,
                           create_supervised_evaluator)
from ignite.metrics import Accuracy, Loss

LOG_INTERVAL = 10


def _flatten_output(logits, target_labels):
    vocab_size = logits.shape[-1]

    flat_logits = logits.view(-1, vocab_size)
    flat_labels = target_labels.view(-1)

    return flat_logits, flat_labels


def episodes_loss(logits, target_labels, **kwargs):
    flat_logits, flat_labels = _flatten_output(logits, target_labels)
    return F.cross_entropy(flat_logits, flat_labels, **kwargs)


def episodes_output_transform(output):
    logits, target_labels = output
    flat_logits, flat_labels = _flatten_output(logits, target_labels)
    return flat_logits, flat_labels


def log_training_loss(trainer):
    loss = trainer.state.output
    epoch = trainer.state.epoch
    iteration = trainer.state.iteration

    if iteration % LOG_INTERVAL == 0:
        print(f"[TRAINING] Epoch {epoch} " f"- Loss: {loss:.2f}")


def log_validation_results(trainer, evaluator, valid_loader):
    epoch = trainer.state.epoch

    evaluator.run(valid_loader)

    metrics = evaluator.state.metrics
    accuracy = metrics['accuracy']
    loss = metrics['loss']

    print(f"[VALIDATION] Epoch {epoch} "
          f"- Avg accuracy: {accuracy:.2f} "
          f"- Avg loss: {loss:.2f}")


def prepare_episodes_batch(batch, device=None, non_blocking=False):
    support_set, targets, labels, target_labels = batch

    inputs = (support_set, targets, labels)
    outputs = target_labels

    if device is not None:
        inputs = (support_set.to(device=device, non_blocking=non_blocking),
                  targets.to(device=device, non_blocking=non_blocking),
                  labels.to(device=device, non_blocking=non_blocking))
        outputs = target_labels.to(device=device, non_blocking=non_blocking)

    return inputs, outputs


def train(model, learning_rate, train_loader, valid_loader, device=None):
    """
    Train the model by getting batches from a data loader.

    Parameters
    ---
    model : nn.Module
        Model which we are training.
    learning_rate : float
        Learning rate at which updates are applied.
    train_loader : torch.DataLoader
        DataLoader for the training set.
    valid_loader : torch.DataLoader
        DataLoader for the validation set.
    device : str
        Where to send data.

    Returns
    ---
    model : nn.Module
        Trained model.
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss = episodes_loss

    trainer = create_supervised_trainer(
        model,
        optimizer,
        loss,
        device=device,
        prepare_batch=prepare_episodes_batch)

    evaluator = create_supervised_evaluator(
        model,
        metrics={
            'accuracy': Accuracy(output_transform=episodes_output_transform),
            'loss': Loss(loss)
        },
        device=device,
        prepare_batch=prepare_episodes_batch)

    trainer.add_event_handler(Events.ITERATION_COMPLETED, log_training_loss)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, log_validation_results,
                              evaluator, valid_loader)

    trainer.run(train_loader, max_epochs=100)

    return model
