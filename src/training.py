from torch import optim
from torch.nn import functional as F

from ignite.engine import Events, create_supervised_trainer

LOG_INTERVAL = 10


def episodes_loss(logits, labels, **kwargs):
    vocab_size = logits.shape[-1]

    flat_logits = logits.view(-1, vocab_size)
    flat_labels = labels.view(-1)

    return F.cross_entropy(flat_logits, flat_labels, **kwargs)


def log_training_loss(trainer):
    loss = trainer.state.output
    epoch = trainer.state.epoch
    iteration = trainer.state.iteration

    if iteration % LOG_INTERVAL == 0:
        print(f"[TRAINING] Epoch {epoch} " f"- Loss: {loss:.2f}")


def prepare_episodes_batch(batch, device=None, non_blocking=False):
    support_set, targets, labels = batch

    inputs = (support_set, targets, labels)
    outputs = labels

    if device is not None:
        outputs = labels.to(device=device, non_blocking=non_blocking)
        inputs = (support_set.to(device=device, non_blocking=non_blocking),
                  targets.to(device=device, non_blocking=non_blocking),
                  outputs)

    return inputs, outputs


def train(model, learning_rate, train_loader, device=None):
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

    trainer.add_event_handler(Events.ITERATION_COMPLETED, log_training_loss)

    trainer.run(train_loader, max_epochs=100)

    return model
