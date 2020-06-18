"""
opt_utils

Optimization utilities for deep generative demixing.

Author: Aaron Berk <aberk@math.ubc.ca>
Copyright © 2020, Aaron Berk, all rights reserved.
Created: 18 June 2020
"""
import numpy as np
import pandas as pd

import torch
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR

from ignite.engine import (
    Events,
    Engine,
    _prepare_batch,
)
from ignite.contrib.handlers import LRScheduler

from util import Logger


def create_lr_finder_engine(
    network, optimizer, criterion, device=None, non_blocking=False
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def update_fn(engine, batch):
        network.train()
        optimizer.zero_grad()
        Xb, yb = _prepare_batch(batch, device=device, non_blocking=non_blocking)
        Xb = Xb.to(device)
        x_recon, mu, log_var = network(Xb)
        loss = criterion(x_recon, Xb, mu, log_var)
        loss.backward()
        optimizer.step()
        return loss.item()

    lr_finder_engine = Engine(update_fn)
    return lr_finder_engine


def create_lr_finder(
    model,
    criterion,
    optim_fn=optim.SGD,
    create_engine=None,
    lr_init=1e-11,
    lr_final=10,
    optim_fn_kwargs=None,
    device=None,
    non_blocking=False,
):
    """
    create_lr_finder(
        model,
        criterion,
        optim_fn=optim.SGD,
        create_engine=None,
        lr_init=1e-11,
        lr_final=10,
        optim_fn_kwargs=None,
        device=None,
    )

    Parameters
    ----------
    model : nn.Module
    criterion : nn.Loss
    optim_fn : torch.optim instance
        Default: optim.SGD
    lr_init : float
    lr_final : float
    optim_fn_kwargs : dict (optional)
    device : torch.device

    Returns
    -------
    find_lr : callable

    Example
    -------
    >>> model = nn.Sequential(nn.Linear(5, 2), nn.ReLU(), nn.Linear(2, 2))
    >>> model_parameter = next(model.parameters())
    >>> # initial value for model_parameter:
    >>> print(model_parameter)
    >>> ## <some tensor>
    >>> criterion = nn.CrossEntropyLoss()
    >>> find_lr = create_lr_finder(model, criterion)
    >>> # plotting does not require GUI
    >>> output = find_lr(loader, plot_fpath="./lr_finder_plot.pdf")
    >>> # the original model's parameters are not modified!
    >>> print(model_parameter)
    >>> ## <the same tensor>
    >>> print(output.keys())
    >>> ## ('lr', 'batch_loss')
    >>> print(len(output["lr"]))
    >>> ## <number of iterations>

    Notes
    -----
    See this article
      https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
    for what might be a better implementation: exponential smoothing and runs
    over a single epoch only. Maybe look at this one too:
      https://forums.fast.ai/t/automated-learning-rate-suggester/44199/15
    which talks about going the final step and choosing an lr automagically.
    """
    from copy import deepcopy

    # Old code:
    # new_model = deepcopy(model)
    if hasattr(model, "_args"):
        new_model = type(model)(*model._args)
    else:
        new_model = deepcopy(model)
    if create_engine is None:
        create_engine = create_lr_finder_engine
    if optim_fn_kwargs is None:
        optim_fn_kwargs = {}
    elif isinstance(optim_fn_kwargs, dict):
        optim_fn_kwargs = {
            key: value for key, value in optim_fn_kwargs.items() if key != "lr"
        }
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = optim_fn(new_model.parameters(), lr=lr_init, **optim_fn_kwargs)

    lr_finder = create_engine(
        new_model,
        optimizer,
        criterion,
        device=device,
        non_blocking=non_blocking,
    )
    exp_scheduler = ExponentialLR(optimizer, gamma=1.1)
    scheduler = LRScheduler(exp_scheduler, save_history=True)
    lr_finder.add_event_handler(Events.ITERATION_COMPLETED, scheduler)
    logger = Logger()

    @lr_finder.on(Events.ITERATION_STARTED)
    def log_lr(engine):
        logger("lr", scheduler.get_param())

    @lr_finder.on(Events.ITERATION_COMPLETED)
    def log_batch_loss(engine):
        logger("batch_loss", engine.state.output)

    @lr_finder.on(Events.ITERATION_COMPLETED)
    def terminate_maybe(engine):
        loss_upper_bound = logger["batch_loss"][0] * 100
        if engine.state.output >= loss_upper_bound:
            engine.terminate()
            return
        if scheduler.get_param() > lr_final:
            engine.terminate()
            return

    @lr_finder.on(Events.COMPLETED)
    def attach_logger(engine):
        if not hasattr(engine.state, "logger"):
            setattr(engine.state, "logger", logger)

    def _get_smoothed_data(output, lr_min, lr_max):
        df = pd.DataFrame(output)
        df["log_lr"] = np.log(df.lr.values)
        df["log_loss"] = np.log(df.batch_loss.values)
        smoothed_log_loss = (
            df.set_index("log_lr")["log_loss"]
            .rolling(10, center=True)
            .mean()
            .reset_index()
        )
        df["lr_smooth"] = np.exp(smoothed_log_loss.log_lr)
        df["batch_loss_smooth"] = np.exp(smoothed_log_loss.log_loss)
        df = df.dropna()
        df = df.loc[(df.lr >= lr_min) & (df.lr <= lr_max)]
        return df

    def _plot_helper(plot_fpath, df, lr_min, lr_max, figsize=None):
        import matplotlib.pyplot as plt

        if figsize is None:
            figsize = (8, 5)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.plot(df.lr, df.batch_loss, label="unsmoothed")
        ax.plot(df.lr_smooth, df.batch_loss_smooth, label="smooth")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lr_min, lr_max)
        ax.set_xlabel("batch loss")
        ax.set_ylabel("lr value")
        ax.legend()
        fig.savefig(plot_fpath)
        del fig, ax
        return

    def _auto_lr_finder(output):
        from scipy.ndimage import gaussian_filter1d

        lr_vec = np.array(output["lr"])
        loss_vec = np.array(output["batch_loss"])
        idx = np.argmin(loss_vec)
        lr_centre = lr_vec[idx]
        lr_min = np.maximum(lr_centre / 100, np.min(lr_vec))
        lr_max = np.minimum(lr_centre * 1000, np.max(lr_vec))

        lr_values = lr_vec[(lr_vec >= lr_min) & (lr_vec <= lr_max)]
        batch_loss = loss_vec[(lr_vec >= lr_min) & (lr_vec <= lr_max)]

        batch_loss_sm = gaussian_filter1d(batch_loss, 1)
        d_batch_loss_sm = gaussian_filter1d(batch_loss, 1, order=1)

        idx_min = np.argmin(batch_loss_sm)
        idx_dec = np.argmin(d_batch_loss_sm[:idx_min])
        lr_star = lr_values[idx_dec]

        if lr_star > 1:
            print("warning: found lr_star > 1. returning 1e-2")
            lr_star = 1e-2
        return lr_star

    def find_lr(dataloader, max_epochs=100, plot_fpath=None, figsize=None):
        """
        find_lr(dataloader, max_epochs=100, plot_fpath=False)

        Parameters
        ----------
        dataloader: torch.utils.data.DataLoader
            dataloader
        max_epochs: int
            upper bound on number of epochs for which to run
        plot_fpath: string
            location of saved plot

        Returns
        -------
        output : dict
            Has keys 'lr' and 'batch_loss'.

        """
        final_state = lr_finder.run(dataloader, max_epochs)

        output = final_state.logger.log
        if isinstance(plot_fpath, str):
            lr_vec = output["lr"]
            loss_vec = output["batch_loss"]
            idx = np.argmin(loss_vec)
            lr_centre = lr_vec[idx]
            lr_min = np.maximum(lr_centre / 100, np.min(lr_vec))
            lr_max = np.minimum(lr_centre * 1000, np.max(lr_vec))
            df = _get_smoothed_data(output, lr_min, lr_max)
            _plot_helper(plot_fpath, df, lr_min, lr_max, figsize=figsize)

        lr_star = _auto_lr_finder(output)
        return lr_star

    return find_lr


# # opt_utils.py ends here
