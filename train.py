"""
train

Methods for training models.

Author: Aaron Berk <aberk@math.ubc.ca>
Copyright © 2020, Aaron Berk, all rights reserved.
Created: 15 June 2020

"""
import pdb
import logging
import torch
from torch import nn, optim
import torch.nn.functional as F
from ignite.engine import Engine, Events, _prepare_batch
from ignite.metrics import Loss, MeanSquaredError

# from ignite.handlers import ModelCheckpoint

from viz import create_save_image_callback
from util import Logger, create_yb_to_one_hot


def get_default_autoencoder_loss(lamda=None):
    if lamda is None:
        lamda = 1.0

    def loss_fn(x_recon, x_true, mu, log_var):
        try:
            BCE = F.binary_cross_entropy(x_recon, x_true, reduction="sum")
        except RuntimeError as rte:
            import sys

            print(rte.args[0])
            print(sys.exc_info())
            pdb.set_trace()
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return BCE + lamda * KLD

    return loss_fn


def create_autoencoder_evaluator(eval_step, metrics=None):
    evaluator = Engine(eval_step)
    if metrics is None:
        return evaluator
    for metric_name, metric in metrics.items():
        metric.attach(evaluator, metric_name)
    return evaluator


def loss_eval_output_transform(output):
    Xr, Xb, yb, mu, log_var = output
    ret = (Xr, Xb, {"mu": mu, "log_var": log_var})
    return ret


def create_log_handler(trainer):
    logger = Logger()

    def log_metrics(engine, phase):
        print(
            f"Epoch {trainer.state.epoch} {phase} loss: {engine.state.metrics['loss']}"
        )
        for metric_name, metric_value in engine.state.metrics.items():
            logger(f"{phase}_{metric_name}", metric_value)

    return log_metrics, logger


def create_vae_train_step(
    model, optimizer, criterion, device=None, non_blocking=False
):
    device = model.device
    if criterion is None:
        criterion = get_default_autoencoder_loss()

    def train_step(engine, batch):
        model.train()
        optimizer.zero_grad()
        Xb, yb = _prepare_batch(batch, device=device, non_blocking=non_blocking)
        Xb = Xb.to(device)
        x_recon, mu, log_var = model(Xb)
        loss = criterion(x_recon, Xb, mu, log_var)
        loss.backward()
        optimizer.step()
        return loss.item()

    return train_step


def create_vae_eval_step(model, device=None, non_blocking=False):
    device = model.device

    def eval_step(engine, batch):
        model.eval()
        Xb, yb = _prepare_batch(batch, device=device, non_blocking=non_blocking)
        with torch.no_grad():
            Xr, mu, log_var = model(Xb)
        return Xr, Xb, yb, mu, log_var

    return eval_step


def create_vae_engines(
    model,
    optimizer,
    criterion=None,
    metrics=None,
    device=None,
    non_blocking=False,
    fig_dir=None,
    unflatten=None,
):

    device = model.device
    if criterion is None:
        criterion = get_default_autoencoder_loss()

    train_step = create_vae_train_step(
        model, optimizer, criterion, device=device, non_blocking=non_blocking
    )
    eval_step = create_vae_eval_step(
        model, device=device, non_blocking=non_blocking
    )

    if metrics is None:
        metrics = {}
    metrics.setdefault(
        "loss", Loss(criterion, output_transform=loss_eval_output_transform),
    )
    metrics.setdefault(
        "mse", MeanSquaredError(output_transform=lambda x: x[:2])
    )
    trainer = Engine(train_step)
    evaluator = create_autoencoder_evaluator(eval_step, metrics=metrics)

    save_image_callback = create_save_image_callback(
        fig_dir, unflatten=unflatten
    )

    def _epoch_getter():
        return trainer.state.__dict__.get("epoch", None)

    evaluator.add_event_handler(
        Events.ITERATION_COMPLETED(once=1),
        save_image_callback,
        epoch=_epoch_getter,
    )

    val_log_handler, val_logger = create_log_handler(trainer)

    return trainer, evaluator, val_log_handler, val_logger


def create_cvae_train_step(
    model,
    optimizer,
    yb_to_one_hot,
    criterion=None,
    device=None,
    non_blocking=False,
):
    device = model.device
    if criterion is None:
        criterion = get_default_autoencoder_loss()

    def train_step(engine, batch):
        model.train()
        optimizer.zero_grad()
        Xb, yb = _prepare_batch(batch, device=device, non_blocking=non_blocking)
        Xb = Xb.to(device)
        y_one_hot = yb_to_one_hot(yb).to(device)
        x_recon, mu, log_var = model((Xb, y_one_hot))
        loss = criterion(x_recon, Xb, mu, log_var)
        loss.backward()
        optimizer.step()
        return loss.item()

    return train_step


def create_cvae_eval_step(
    model, yb_to_one_hot, device=None, non_blocking=False
):
    device = model.device

    def eval_step(engine, batch):
        model.eval()
        Xb, yb = _prepare_batch(batch, device=device, non_blocking=non_blocking)
        with torch.no_grad():
            y_one_hot = yb_to_one_hot(yb).to(device)
            Xr, mu, log_var = model((Xb, y_one_hot))
        return Xr, Xb, yb, mu, log_var

    return eval_step


def create_cvae_engines(
    model: nn.Module,
    optimizer,
    classes,
    criterion=None,
    metrics=None,
    device=None,
    non_blocking=False,
    fig_dir=None,
    unflatten=None,
    batch_size=None,
):

    device = model.device
    if criterion is None:
        criterion = get_default_autoencoder_loss()

    if batch_size is None:
        print("Assuming batch_size = 128.")
        batch_size = 128

    yb_to_one_hot = create_yb_to_one_hot(batch_size, classes)

    train_step = create_cvae_train_step(
        model,
        optimizer,
        yb_to_one_hot,
        criterion=criterion,
        device=device,
        non_blocking=non_blocking,
    )
    eval_step = create_cvae_eval_step(
        model, yb_to_one_hot, device=device, non_blocking=non_blocking
    )

    if metrics is None:
        metrics = {}
    metrics.setdefault(
        "loss", Loss(criterion, output_transform=loss_eval_output_transform),
    )
    metrics.setdefault(
        "mse", MeanSquaredError(output_transform=lambda x: x[:2])
    )
    trainer = Engine(train_step)
    evaluator = create_autoencoder_evaluator(eval_step, metrics=metrics)

    save_image_callback = create_save_image_callback(
        fig_dir, unflatten=unflatten
    )

    def _epoch_getter():
        return trainer.state.__dict__.get("epoch", None)

    evaluator.add_event_handler(
        Events.ITERATION_COMPLETED(once=1),
        save_image_callback,
        epoch=_epoch_getter,
    )

    val_log_handler, val_logger = create_log_handler(trainer)

    return trainer, evaluator, val_log_handler, val_logger


def create_default_engines_from_steps(
    train_step,
    eval_step,
    criterion=None,
    metrics=None,
    fig_dir=None,
    unflatten=None,
):
    """
    create_default_engines_from_steps(
        train_step,
        eval_step,
        criterion=None,
        metrics=None,
        fig_dir=None,
        unflatten=None,
    )

    Parameters
    ----------
    train_step : callable
        The update function for the trainer
    eval_step : callable
        The update function for the evaluator
    criterion : nn.Loss (optional)
        Note: if criterion is not passed, then validation loss will not be
        tracked by ignite, unless passed via metrics.
    metrics : dict (optional)
    fig_dir : string (optional)
    unflatten : tuple (optional)
    
    Returns
    -------
    trainer : ignite Engine
    evaluator : ignite Engine
    val_log_handler : ignite handler
        To be used with add_evaluation and some dataloaders, in order to track
        progress on a validation set during training.
    val_logger : util.Logger
        Object containing the validation metric data from training.
    """
    if metrics is None:
        metrics = {}
    if criterion is not None:
        metrics.setdefault(
            "loss",
            Loss(criterion, output_transform=loss_eval_output_transform),
        )
    metrics.setdefault(
        "mse", MeanSquaredError(output_transform=lambda x: x[:2])
    )
    trainer = Engine(train_step)
    evaluator = create_autoencoder_evaluator(eval_step, metrics=metrics)

    save_image_callback = create_save_image_callback(
        fig_dir, unflatten=unflatten
    )

    def _epoch_getter():
        return trainer.state.__dict__.get("epoch", None)

    evaluator.add_event_handler(
        Events.ITERATION_COMPLETED(once=1),
        save_image_callback,
        epoch=_epoch_getter,
    )

    val_log_handler, val_logger = create_log_handler(trainer)

    return trainer, evaluator, val_log_handler, val_logger


def set_evaluator_phase(engine, phase):
    engine.state.__dict__["phase"] = phase


def run_evaluator(evaluator: Engine, log_handler, loader, phase):
    with evaluator.add_event_handler(
        Events.EPOCH_STARTED, set_evaluator_phase, phase
    ):
        with evaluator.add_event_handler(
            Events.EPOCH_COMPLETED, log_handler, phase
        ):
            evaluator.run(loader)
    return


def add_evaluation(trainer, evaluator, log_handler, val_loaders):
    @trainer.on(Events.EPOCH_COMPLETED)
    def run_evaluation(engine):
        for phase, loader in val_loaders.items():
            run_evaluator(evaluator, log_handler, loader, phase)

    return trainer


criteria = {"default": get_default_autoencoder_loss}
optimizers = {"SGD": optim.SGD, "Adam": optim.Adam}

# # train.py ends here
