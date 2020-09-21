"""
gd_numerics_demix_with_trained

Runs demixing_problem_two_network for two generators that were trained with
gd_numerics_two_network.run.

Author: Aaron Berk <aberk@math.ubc.ca>
Copyright © 2020, Aaron Berk, all rights reserved.
Created: 20 September 2020
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from gd_numerics_two_network import run
from demix import demixing_problem_two_network
from util import search_for_tstamp_directory, get_tstamp


tstamp_pairs = [["20200914-105758-237836", "20200914-105759-069972"]]


def load(tstamps):
    """
    Returns
    -------
    output
    """
    output = run(train=False, load_from_tstamp=True, tstamps=tstamps)

    return output


def get_images(output, phase="test"):
    """
    Parameters
    ----------
    output: tuple
        The tuple returned by load.
    phase: str
        train, val or test.

    Returns
    -------
    imgs : dict
        Sampled from the two dataloaders for phase `phase`; dict has keys equal
        to the digit_class.
    """
    args = output[0]
    if not isinstance(args, dict):
        emsg = f"Expected dict for args, where each key is a digit_class but got {type(args)}"
        raise TypeError(emsg)
    num_digit_classes = len(args)
    digit_classes = list(args.keys())

    dataloaders = output[1]
    imgs = {}
    for dc, loaders in dataloaders.items():
        loader = loaders.get(phase, None)
        if loader is None:
            emsg = f"phase {phase} not found in dataloaders[{dc}]"
            raise KeyError(emsg)
        # get a random image
        for batch in loader:
            imgs[dc] = batch[0][0]
    return imgs


def get_networks(output):
    """
    Parameters
    ----------
    output: tuple
        The tuple returned by `load`

    Returns
    -------
    networks : dict
        networks is a dict of trained networks with keys equal to tstamps that
        were passed to `load`.

    """
    return output[2]


def setup_demixing(output, phase="test", seed=None):
    """
    parameters

    Returns
    -------
    imgs : dict
        Values in the dict should be torch.Tensors. Probably with size [1, X, Y]?
    networks : dict
        Values in the dict should be nn.Module objects
    A : torch.Tensor
        m x n measurement matrix with N(0, 1/m) entries where m is the number of
        elements in the first Tensor in imgs, and n is the number of elements in
        the second.
    """
    imgs = get_images(output, phase=phase)
    img_classes = list(imgs.keys())
    m = np.prod(imgs[img_classes[0]].shape)
    n = np.prod(imgs[img_classes[1]].shape)
    if seed is not None:
        torch.manual_seed(seed)
    A = torch.randn(m, n) / m ** 0.5

    networks = get_networks(output)
    return imgs, networks, A


def run_demixing(**kwargs):
    """
    Parameters
    ----------
    tstamps : list
    num_iter : int
    clamp : bool
    device : str
    phase : str
    seed : int (optional)

    Returns
    -------
    results : dict
        For a list of keys and their descriptions, see demixing_problem_two_network
    """
    tstamps = kwargs.get("tstamps", None)
    num_iter = kwargs.get("num_iter", 1000)
    clamp = kwargs.get("clamp", True)
    device = kwargs.get(
        "device", "cuda" if torch.cuda.is_available() else "cpu"
    )
    phase = kwargs.get("phase", "test")
    seed = kwargs.get("seed", None)

    if not isinstance(tstamps, (list, tuple)):
        raise TypeError(f"Unexpected input for tstamps, got {tstamps}")

    output = load(tstamps)
    imgs, networks, A = setup_demixing(output, phase=phase, seed=seed)

    results = demixing_problem_two_network(
        networks, imgs, A, num_iter=num_iter, clamp=clamp, device=device
    )
    results["phase"] = phase
    results["seed"] = seed
    results["tstamps"] = tstamps
    results["clamp"] = clamp
    results["num_iter"] = num_iter
    results["A"] = A
    results["networks"] = networks
    results["imgs"] = imgs
    return results


def save_image_dict_plots(image_dict, figsize=(5, 5), **kwargs):
    """Plots the results of run_demixing and saves them.

    Parameters
    ----------
    image_dict: dict of arrays
        all values are assumed to be images that need to be plotted, except for
        the key 'logger', which assumed to be a util.Logger instance with keys
        'iter' and 'loss'.
    figsize: tuple
        figure size for each of the individual figures
        Note: figsize for logger loss plot: 
            (figsize[1] * 2 ** 0.5, figsize[1] / 2 ** 0.5)
    plot_fpath : str
        Must have "{key}" in plot_fpath so that formatter can embed key of
        image_dict into plot name. Default: plot_{key}.pdf
    savefig_kwargs : dict
        Other args to fig.savefig, like dpi.

    Returns
    -------
    out : dict
        Has keys same as image_dict, except for 'logger' which is renamed to
        'loss' (if present). Each key contains a `fig` object.
    """
    plot_fpath = kwargs.pop("plot_fpath", "plot_{key}.pdf")
    kwargs.setdefault("dpi", 300)
    kwargs.setdefault("bbox_inches", "tight")
    kwargs.setdefault("pad_inches", 0.1)

    logger = image_dict.pop("logger", None)
    out = {}
    for key, image in image_dict.items():
        if key == "logger":
            continue
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.imshow(image, cmap="gray")
        ax.axis("off")
        fig.tight_layout()
        plot_fpath_ = plot_fpath.format(key=key)
        print(f"\nWriting plot to\n  {plot_fpath_}")
        fig.savefig(plot_fpath_, **kwargs)

    if logger is not None:
        plt.rcParams["axes.labelsize"] = 14
        plt.rcParams["font.size"] = 14
        plt.rcParams["lines.linewidth"] = 2
        log_figsize = (figsize[1] * 2 ** 0.5, figsize[1] / 2 ** 0.5)
        fig, ax = plt.subplots(1, 1, figsize=log_figsize)
        ax.plot(logger["iter"], logger["loss"])
        ax.set_ylabel("MSE")
        ax.set_xlabel("iter")
        ax.set_yscale("log")
        fig.tight_layout()
        plot_fpath_ = plot_fpath.format(key="loss")
        print(f"\nWriting plot to\n  {plot_fpath_}")
        fig.savefig(plot_fpath_, **kwargs)


def save_results(results, current_tstamp, figsize=(5, 5)):
    """
    Takes results and generates:
    - info.csv : information about the networks and data used in the demixing setup
    - log.csv
    - results.npz
    - results_recovered_plot.pdf
    - results_mixture_plot.pdf
    """
    tstamps = list(results["networks"].keys())
    digit_classes = list(results["imgs"].keys())

    directories = [search_for_tstamp_directory(tstamp) for tstamp in tstamps]
    parent_dir = os.path.commonpath(directories)
    mixture_dir = os.path.join(parent_dir, f"demix_{current_tstamp}")
    if not os.path.exists(mixture_dir):
        os.makedirs(mixture_dir)

    info_fpath = os.path.join(mixture_dir, "info.txt")
    log_fpath = os.path.join(mixture_dir, "log.csv")
    results_fpath = os.path.join(mixture_dir, "results.npz")
    plot_fpath = os.path.join(mixture_dir, "results_{key}_plot.pdf")

    def _fmt_tens(tens):
        return tens.detach().squeeze().cpu().numpy()

    image_dict = {
        "true0": _fmt_tens(results["x0"]),
        "recovered0": _fmt_tens(results["demixed0"]),
        "true1": _fmt_tens(results["x1"]),
        "recovered1": _fmt_tens(results["demixed1"]),
        "mixture": _fmt_tens(results["mixture"]),
        "mixture_pred": _fmt_tens(results["mixture_pred"]),
        "logger": results["logger"],
    }

    err0 = image_dict["true0"] - image_dict["recovered0"]
    err1 = image_dict["true1"] - image_dict["recovered1"]
    err_mixture = image_dict["mixture"] - image_dict["mixture_pred"]
    mse0 = np.linalg.norm(err0) ** 2 / err0.size
    mse1 = np.linalg.norm(err1) ** 2 / err1.size
    mse_mixture = np.linalg.norm(err_mixture) ** 2 / err_mixture.size

    INFO = f"""Results from running demixing_problem_two_network for two generators that were trained with gd_numerics_two_network.run

current_tstamp: {current_tstamp}

tstamps: {tstamps}
digit_classes: {digit_classes}
phase: {results['phase']}

seed: {results['seed']}
A_shape: {results['A'].shape}
clamp: {results['clamp']}
num_iter: {results['num_iter']}

mse_img_0: {mse0:.3e}
mse_img_1: {mse1:.3e}
mse_mixture: {mse_mixture:.3e}
"""
    print(f"Writing info to\n  {info_fpath}")
    with open(info_fpath, "w") as fp:
        fp.write(INFO)
    logger = results["logger"]
    print(f"Writing log to\n  {log_fpath}")
    logger.save(log_fpath)

    arr_dict = {}
    for key in [
        "A",
        "demixed0",
        "demixed1",
        "mixture",
        "mixture_pred",
        "x0",
        "x1",
    ]:
        arr_dict[key] = results[key].detach().squeeze().cpu().numpy()
    print(f"Writing arrays to\n  {results_fpath}")
    np.savez_compressed(results_fpath, **arr_dict)

    save_image_dict_plots(image_dict, figsize=figsize, plot_fpath=plot_fpath)
    return


if __name__ == "__main__":
    current_tstamp = get_tstamp()
    results = run_demixing(tstamps=tstamp_pairs[0], seed=2020)

    save_results(results, current_tstamp)


# # gd_numerics_demix_with_trained.py ends here
