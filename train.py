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
from torch import optim
import torch.nn.functional as F
from ignite.engine import Engine, Events, _prepare_batch
from ignite.metrics import Loss, MeanSquaredError

# from ignite.handlers import ModelCheckpoint

from viz import create_save_image_callback
from util import Logger


def get_default_autoencoder_loss(lamda=None):
    if lamda is None:
        lamda = 1.0

    def loss_fn(x_recon, x_true, mu, log_var):
        try:
            BCE = F.binary_cross_entropy(x_recon, x_true, reduction="mean")
        except RuntimeError:
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
    # pdb.set_trace()
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


def create_autoencoder_engines(
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

    def eval_step(engine, batch):
        model.eval()
        Xb, yb = _prepare_batch(batch, device=device, non_blocking=non_blocking)
        with torch.no_grad():
            Xr, mu, log_var = model(Xb)
        return Xr, Xb, yb, mu, log_var

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
