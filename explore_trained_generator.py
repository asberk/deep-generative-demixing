"""
explore_trained_generator

Working demixing example. Need to refactor and spruce it up.

Author: Aaron Berk <aberk@math.ubc.ca>
Copyright © 2020, Aaron Berk, all rights reserved.
Created: 19 June 2020

"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn, optim

from data import load_data_fns
from util import load_saved_model_by_tstamp


def load_a_trained_model_with_data(tstamp=None):
    if tstamp is None:
        tstamp = "20200619-091719-034209"
    args_df = pd.read_csv(f"./log/{tstamp}/args.csv")
    data_name = args_df.data[0]
    data_kwargs = eval(args_df.data_kwargs[0])

    model = load_saved_model_by_tstamp(tstamp, in_features=784)
    model.eval()

    dataloaders, img_shape, classes = load_data_fns[data_name](data_kwargs)
    return tstamp, args_df, model, dataloaders, img_shape, classes


def _plot_encoding(*encodings, fpath=None, figsize=None):
    if fpath is None:
        fpath = "./encoding_plot.pdf"
    if figsize is None:
        figsize = (8, 6)
    num_encodings = len(encodings)
    fig, ax = plt.subplots(2, 1, figsize=figsize)
    for i in range(num_encodings):
        ax[0].plot(encodings[i][0], label=f"$\\mu_{i}$")
        ax[1].plot(encodings[i][1], label=f"$2\\log\\sigma_{i}$")
    ax[0].legend()
    ax[1].legend()
    plt.tight_layout()
    fig.savefig(fpath)
    del fig, ax
    return


def _plot_interpolation(
    get_image_generator, z0, z1, num_plots=8, fpath=None, figsize=None
):
    if figsize is None:
        figsize = (10, 4)
    fig, ax = plt.subplots(1, num_plots, figsize=figsize)
    for i, image in enumerate(get_image_generator(z0, z1, num_plots)):
        image = image.view(28, 28)
        ax[i].imshow(image, cmap="gray")
        ax[i].axis("off")
    plt.tight_layout()
    fig.savefig("./interpolation_plot.pdf")
    del fig, ax
    return


def get_linear_interpolant(x0, x1, t):
    return x1.mul(t) + x0.mul(1 - t)


def interpolation_example(
    dataloader,
    model,
    encoding_plot_fpath=None,
    interpolation_plot_fpath=None,
    encoding_plot_figsize=None,
    interpolation_plot_figsize=None,
):

    sample0 = dataloader.dataset[0]
    print(sample0[1])
    sample1 = dataloader.dataset[7500]
    print(sample1[1])

    with torch.no_grad():
        enc0 = model.encode(sample0[0])
        enc1 = model.encode(sample1[0])
        z0 = model.reparametrize(*enc0)
        z1 = model.reparametrize(*enc1)

    _plot_encoding(
        enc0, enc1, fpath=encoding_plot_fpath, figsize=encoding_plot_figsize
    )

    def get_image_interpolants(z_start, z_end, num_steps):
        t_vec = np.linspace(0, 1, num_steps)
        model.eval()
        for t in t_vec:
            z_t = get_linear_interpolant(z_start, z_end, t)
            with torch.no_grad():
                dec = model.decode(z_t)
            yield dec

    _plot_interpolation(
        get_image_interpolants,
        z0,
        z1,
        num_plots=8,
        fpath=interpolation_plot_fpath,
        figsize=interpolation_plot_figsize,
    )

    return


def get_two_images_from_dataloader(dataloader):
    dset = dataloader.dataset
    length = len(dset)
    for i in range(length):
        sample0 = dset[i]
        if sample0[1] == 1:
            break
    if i > 0:
        sample1 = dset[i - 1]
        return sample0[0], sample1[0]
    for i in range(length):
        sample1 = dset[i]
        if sample1[1] == 8:
            return sample0[0], sample1[0]
    raise RuntimeError("Could not find a pair of samples labelled 1 and 8.")


def _plot_images(*images):
    fig, ax = plt.subplots(1, len(images), squeeze=False)
    print(ax)
    for i, image in enumerate(images):
        ax[0, i].imshow(image.view(28, 28).detach().numpy(), cmap="gray")
        ax[0, i].axis("off")
    plt.tight_layout()
    plt.show()
    del fig, ax
    return


def demixing_problem(
    model: nn.Module,
    x0: torch.Tensor,
    x1: torch.Tensor,
    Q=None,
    num_iter=1000,
    clamp=True,
):

    x0_shape = x0.shape
    x1_shape = x1.shape
    assert (
        x0_shape == x1_shape
    ), f"Expected x0.shape == x1.shape but found {x0.shape} != {x1.shape}"
    if (Q is not None) and isinstance(Q, torch.Tensor):
        mixture = x0 + torch.matmul(Q, x1.view(-1, 1)).view(*x1_shape)
    else:
        mixture = x0 + x1

    if clamp:
        mixture.clamp_(0.0, 1.0)

    model.eval()

    mixture_params = model.encode(mixture.view(1, -1))
    mixture_encoding = model.reparametrize(*mixture_params)

    # Set requires_grad = False for all model parameters.
    model.requires_grad_(False)

    # For encoding vectors w0 and w1, set requires_grad = True.
    w0 = (
        mixture_encoding.clone()
        .detach()
        .add_(torch.randn_like(mixture_encoding), alpha=0.1)
        .requires_grad_(True)
    )

    w1 = (
        mixture_encoding.clone()
        .detach()
        .add_(torch.randn_like(mixture_encoding), alpha=0.1)
        .requires_grad_(True)
    )

    # Acquire im0 = model.decode(w0), im1 = model.decode(w1)
    #   and compute loss = norm(y - im0 - im1, 2)**2 where
    #   w0.requires_grad = True and w1.requires_grad = True.
    # Then after we call loss.backward, we should have updates for
    #   w0 and w1. Just gotta' pass w0 and w1 to the optimizer.

    criterion = nn.MSELoss()
    optimizer = optim.Adam([w0, w1], lr=1e-2)

    for i in range(num_iter):
        demixed0 = model.decode(w0)
        demixed1 = model.decode(w1)
        optimizer.zero_grad()
        loss = criterion(demixed0 + demixed1, mixture)
        loss.backward()
        optimizer.step()
    return demixed0, w0, demixed1, w1, mixture, mixture_encoding, optimizer


if __name__ == "__main__":
    (
        tstamp,
        args_df,
        model,
        dataloaders,
        img_shape,
        classes,
    ) = load_a_trained_model_with_data()

    # fig_dir = os.path.join("./fig/", tstamp)
    # encoding_plot_fpath = os.path.join(fig_dir, "encoding_plot.pdf")
    # interpolation_plot_fpath = os.path.join(fig_dir, "interpolation_plot.pdf")
    # interpolation_example(
    #     dataloaders["train"],
    #     model,
    #     encoding_plot_fpath,
    #     interpolation_plot_fpath,
    # )

    x0, x1 = get_two_images_from_dataloader(dataloaders["train"])
    (
        demixed0,
        w0,
        demixed1,
        w1,
        mixture,
        mixture_encoding,
        optimizer,
    ) = demixing_problem(model, x0, x1, clamp=True)

    _plot_images(x0, x1, mixture, demixed0, demixed1)


# # explore_trained_generator.py ends here
