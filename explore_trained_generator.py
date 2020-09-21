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

from data import load_data_fns
from demix import (
    demixing_problem_one_network,
    multi_demixing_problem_one_network,
)
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


def get_n_images_from_dataloader(dataloader, classes, seed=None):
    dset = dataloader.dataset
    length = len(dset)
    if seed is not None:
        np.random.seed(seed)
    if isinstance(classes, torch.Tensor):
        classes = classes.detach().cpu().numpy()
    indices = np.random.choice(length, size=length, replace=False)
    images = {}
    for i in indices:
        sample = dset[i]
        image, label = sample
        if (label in classes) and (images.get(label, None) is None):
            images[label] = image
        if len(images) == classes.size:
            return images
    raise RuntimeError(
        f"Could not find an image for each label in classes. "
        f"Found only: {tuple(images.keys())}."
    )


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


def simple_demixing_example(num_iter=1000, clamp=True, seed=2020):
    (
        tstamp,
        args_df,
        model,
        dataloaders,
        img_shape,
        classes,
    ) = load_a_trained_model_with_data()

    images = get_n_images_from_dataloader(
        dataloaders["test"], classes, seed=seed
    )
    images_ = list(images.values())

    (
        demixed0,
        w0,
        demixed1,
        w1,
        mixture,
        mixture_encoding,
        optimizer,
        logger,
    ) = demixing_problem_one_network(
        model, *images_, num_iter=num_iter, clamp=clamp
    )

    _plot_images(*images_, mixture, demixed0, demixed1)

    history = logger.to_df()

    results = {
        "tstamp": tstamp,
        "args_df": args_df,
        "model": model,
        "dataloaders": dataloaders,
        "images": images,
        "demixed": [demixed0, demixed1],
        "Wi": [w0, w1],
        "mixture": mixture,
        "mixture_encoding": mixture_encoding,
        "optimizer": optimizer,
        "history": history,
    }

    return results


def three_class_demixing_example(num_iter=2000, clamp=True, seed=2020):
    tstamp = "20200626-140403-564632"
    (
        tstamp,
        args_df,
        model,
        dataloaders,
        img_shape,
        classes,
    ) = load_a_trained_model_with_data(tstamp)

    images = get_n_images_from_dataloader(
        dataloaders["test"], classes, seed=seed
    )
    multi_demix_output = multi_demixing_problem_one_network(
        model, images, num_iter=num_iter, clamp=clamp
    )
    (
        demixed,
        Wi,
        mixture,
        mixture_encoding,
        optimizer,
        logger,
    ) = multi_demix_output
    history = logger.to_df()
    print(history.head(5))
    print()
    print(history.tail(5))

    _plot_images(*list(images.values()), mixture, *demixed)
    results = {
        "tstamp": tstamp,
        "args_df": args_df,
        "model": model,
        "dataloaders": dataloaders,
        "images": images,
        "demixed": demixed,
        "Wi": Wi,
        "mixture": mixture,
        "mixture_encoding": mixture_encoding,
        "optimizer": optimizer,
        "history": history,
    }
    return results


def default_interpolation_example():
    (
        tstamp,
        args_df,
        model,
        dataloaders,
        img_shape,
        classes,
    ) = load_a_trained_model_with_data()

    fig_dir = os.path.join("./fig/", tstamp)
    encoding_plot_fpath = os.path.join(fig_dir, "encoding_plot.pdf")
    interpolation_plot_fpath = os.path.join(fig_dir, "interpolation_plot.pdf")
    interpolation_example(
        dataloaders["test"],
        model,
        encoding_plot_fpath,
        interpolation_plot_fpath,
    )
    return


if __name__ == "__main__":

    results = simple_demixing_example(num_iter=100)
    # results = three_class_demixing_example()


# # explore_trained_generator.py ends here
