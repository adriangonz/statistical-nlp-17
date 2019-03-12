from torch import optim
from torch.nn import functional as F

from ignite.engine import Events, create_supervised_trainer

LOG_INTERVAL = 10


def log_training_loss(trainer):
    loss = trainer.state.output
    epoch = trainer.state.epoch
    iteration = trainer.state.iteration

    #  if iteration % LOG_INTERVAL == 0:
    print(f"[TRAINING] Epoch {epoch} " f"- Loss: {loss:.2f}")


def prepare_episodes_batch(batch, device=None, non_blocking=False):
    support_set, targets, labels = batch

    inputs = (support_set, targets)
    outputs = labels

    if device is not None:
        inputs = (support_set.to(device=device, non_blocking=non_blocking),
                  targets.to(device=device, non_blocking=non_blocking))
        outputs = labels.to(device=device, non_blocking=non_blocking)

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
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=(0.1, 0.001),
        weight_decay=1e-7)
    loss = F.cross_entropy

    trainer = create_supervised_trainer(
        model,
        optimizer,
        loss,
        device=device,
        prepare_batch=prepare_episodes_batch)

    trainer.add_event_handler(Events.ITERATION_COMPLETED, log_training_loss)

    trainer.run(train_loader, max_epochs=100)

    return model
