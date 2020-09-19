"""
viz

Some tools to plot mnist digits

Author: Aaron Berk <aberk@math.ubc.ca>
Copyright © 2020, Aaron Berk, all rights reserved.
Created: 15 June 2020
"""
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision.utils import save_image

from util import get_tstamp


def plot_random_images(dataset, k=8, nr=2, figsize=None):
    indices = np.random.choice(len(dataset), size=k, replace=False)
    nc = (k // nr) if ((k % nr) == 0) else (k // nr + 1)
    if figsize is None:
        figsize = (2 * nc, 2 * nr)
    fig, ax = plt.subplots(nr, nc, figsize=figsize)
    ax = ax.ravel()
    for i, idx in enumerate(indices):
        img, lab = dataset[idx]
        ax[i].imshow(img.squeeze().numpy())
        ax[i].set_title(f"label: {lab}")
        ax[i].axis("off")
    plt.tight_layout()
    plt.show()
    del fig, ax
    return


def create_save_image_callback(
    save_dir=None, fname_pattern=None, unflatten=None
):
    if save_dir is None:
        save_dir = "./fig/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if fname_pattern is None:
        fname_pattern = "img{epoch}{phase}_{tstamp}.jpg"

    if (unflatten is None) or (unflatten is False):
        pass
    else:
        if isinstance(unflatten, np.int):
            unflatten = (unflatten, unflatten)
        assert isinstance(unflatten, (tuple, list))
        if len(unflatten) == 2:
            unflatten = (1, *unflatten)
        assert (
            len(unflatten) == 3
        ), f"Expected unflatten to be a tuple of length 2 or 3."

    def save_image_callback(engine, epoch=None):
        Xr, Xb, *_ = engine.state.output
        tstamp = get_tstamp()
        phase = engine.state.__dict__.get("phase", None)
        epoch_tag = f"_{epoch()}" if callable(epoch) else ""
        phase_tag = f"_{phase}" if isinstance(phase, str) else ""
        fpath = os.path.join(
            save_dir,
            fname_pattern.format(
                epoch=epoch_tag, phase=phase_tag, tstamp=tstamp,
            ),
        )
        Xrb = torch.cat((Xr, Xb), dim=0)
        if isinstance(unflatten, (tuple, list)):
            Xrb = Xrb.view(Xrb.size(0), *unflatten)
        save_image(Xrb, fpath)
        return

    return save_image_callback


def plot_batch(batch, nr=2):
    X, y = batch
    nc = X.size(0) // nr if ((X.size(0) % nr) == 0) else X.size(0) // nr + 1
    fig, ax = plt.subplots(nr=nr, nc=nc, figsize=(2 * nc, 2 * nr))
    ax = ax.ravel()
    for i in range(X.size(0)):
        img = X[i].squeeze().numpy()
        label = y[i].item()
        ax[i].imshow(img)
        ax[i].set_title(f"label: {label}")
        ax[i].axis("off")
    plt.tight_layout()
    plt.show()
    del fig, ax
    return


def plot_val_log(val_log, figsize=None, title=None):
    if figsize is None:
        figsize = (10, 4)
    plt.rcParams["font.size"] = 16
    plt.rcParams["axes.labelsize"] = 16
    plt.rcParams["lines.linewidth"] = 2
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    ax[0].plot(
        val_log.train_eval_epoch, val_log.train_eval_loss, label="train_eval",
    )
    ax[0].plot(val_log.val_epoch, val_log.val_loss, label="val")
    ax[0].legend()
    t1 = "loss" if title is None else f"{title} loss"
    ax[0].set_title(t1, size=12)
    ax[1].plot(
        val_log.train_eval_epoch, val_log.train_eval_mse, label="train_eval",
    )
    ax[1].plot(val_log.val_epoch, val_log.val_mse, label="val")
    ax[1].legend()
    t2 = "mse" if title is None else f"{title} mse"
    ax[1].set_title(t2, size=12)
    return fig, ax


# # viz.py ends here
