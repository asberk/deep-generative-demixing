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
    results["A"] = A
    results["networks"] = networks
    results["imgs"] = imgs
    return results


def plot_results(results, figsize=(10, 5)):
    """Plots the results of run_demixing.


    Parameters
    ----------
    results : dict
        Output of run_demixing

    """
    demixed0 = results["demixed0"].detach().squeeze().cpu().numpy()
    demixed1 = results["demixed1"].detach().squeeze().cpu().numpy()
    logger = results["logger"]
    mixture = results["mixture"].detach().squeeze().cpu().numpy()
    mixture_pred = results["mixture_pred"].detach().squeeze().cpu().numpy()
    x0 = results["x0"].detach().squeeze().cpu().numpy()
    x1 = results["x1"].detach().squeeze().cpu().numpy()

    fig, ax = plt.subplots(2, 3, figsize=figsize)
    ax[0, 0].imshow(x0, cmap="gray")
    ax[0, 1].imshow(demixed0, cmap="gray")
    ax[1, 0].imshow(x1, cmap="gray")
    ax[1, 1].imshow(demixed1, cmap="gray")
    ax[0, 2].plot(logger["iter"], logger["loss"])
    ax[0, 2].set_ylabel("MSE")
    ax[0, 2].set_xlabel("iter")
    ax[0, 2].set_yscale("log")

    titles = ["true", "recovered"]
    for j in range(2):
        ax[0, j].set_title(titles[j])

    for i in range(2):
        for j in range(2):
            ax[i, j].axis("off")
    ax[1, 2].axis("off")

    mse_mixture = np.linalg.norm(mixture - mixture_pred) ** 2 / mixture.size
    print(f"mse_mixture: {mse_mixture:.4f}")

    plt.tight_layout()

    return fig, ax


def save_results(results, current_tstamp):
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
    plot_fpath = os.path.join(mixture_dir, "results_plot.pdf")

    INFO = f"""Results from running demixing_problem_two_network for two generators that were trained with gd_numerics_two_network.run

current_tstamp: {current_tstamp}
tstamps: {tstamps}
digit_classes: {digit_classes}
"""
    print(f"Writing info to\n  {info_fpath}")
    with open(info_fpath, "w") as fp:
        fp.write(INFO)
    logger = results["logger"]
    print(f"Writing log to\n  {log_fpath}")
    logger.save(log_fpath)

    arr_dict = {}
    for key in ["demixed0", "demixed1", "mixture", "mixture_pred", "x0", "x1"]:
        arr_dict[key] = results[key].detach().squeeze().cpu().numpy()
    print(f"Writing arrays to\n  {results_fpath}")
    np.savez_compressed(results_fpath, **arr_dict)

    fig, ax = plot_results(results)
    print(f"Writing plot to\n  {plot_fpath}")
    fig.savefig(plot_fpath, dpi=300)
    return


if __name__ == "__main__":
    current_tstamp = get_tstamp()
    results = run_demixing(tstamps=tstamp_pairs[0], seed=2020)

    save_results(results, current_tstamp)


# # gd_numerics_demix_with_trained.py ends here
