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


def create_save_image_callback(save_dir=None, fname_pattern=None):
    if save_dir is None:
        save_dir = "./fig/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if fname_pattern is None:
        fname_pattern = "{img_type}{epoch}{phase}_{tstamp}.jpg"

    def save_image_callback(engine, epoch=None):
        Xr, Xb, *_ = engine.state.output
        tstamp = get_tstamp()
        phase = engine.state.__dict__.get("phase", None)
        epoch_tag = f"_{epoch()}" if callable(epoch) else ""
        phase_tag = f"_{phase}" if isinstance(phase, str) else ""
        recon_fpath = os.path.join(
            save_dir,
            fname_pattern.format(
                img_type="recon",
                epoch=epoch_tag,
                phase=phase_tag,
                tstamp=tstamp,
            ),
        )
        batch_fpath = os.path.join(
            save_dir,
            fname_pattern.format(
                img_type="batch",
                epoch=epoch_tag,
                phase=phase_tag,
                tstamp=tstamp,
            ),
        )
        save_image(Xr, recon_fpath)
        save_image(Xb, batch_fpath)
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


# # viz.py ends here
