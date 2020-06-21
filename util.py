import os
from datetime import datetime
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from torch import nn


class Logger:
    def __init__(self):
        self.log = defaultdict(list)

    def __call__(self, key, value):
        self.log[key].append(value)

    def __repr__(self):
        keys = ", ".join(f"{key}" for key in self.log.keys())
        return f"Logger({keys})"

    def __getitem__(self, idx):
        return self.log[idx]

    def to_df(self):
        return pd.DataFrame(self.log)

    def save(self, filepath, **kwargs):
        """
        save(self, filepath, **kwargs)

        Parameters
        ----------
        filepath: string
        kwargs: dict
            keyword arguments to pd.DataFrame.to_csv
        """
        self.to_df().to_csv(filepath, **kwargs)
        print(f"Saved log to\n  {filepath}")

    def clear(self):
        self.log = defaultdict(list)


def get_tstamp():
    tstamp = (
        datetime.now()
        .isoformat()
        .replace("-", "")
        .replace("T", "-")
        .replace(":", "")
        .replace(".", "-")
    )
    return tstamp


def save_model(
    model,
    model_name=None,
    epoch=None,
    score_name=None,
    score_value=None,
    tstamp=None,
    save_dir=None,
):
    """
    save_model(
        model,
        model_name=None,
        epoch=None,
        score_name=None,
        score_value=None,
        tstamp=None,
        save_dir=None,
    )

    Filename format:
        {save_dir}/{model_name}_epoch{epoch}_{score_name}{score_value}_{tstamp}.pth

    Parameters
    ----------
    model : nn.Module
    model_name : string (optional)
    epoch : int or string (optional)
    score_name : string (optional)
    score_value : scalar or string (optional)
    tstamp : string (optional)
    save_dir : string (optional)
    """
    if model_name is None:
        model_name = "model"
    fname = f"{model_name}"
    if isinstance(epoch, (np.int, str)):
        fname += f"_epoch{epoch}"
    if np.isscalar(score_value):
        score_value = f"{score_value}"
    if isinstance(score_value, str):
        if score_name is None:
            score_name = "score"
        fname += f"_{score_name}{score_value}"
    if tstamp is None:
        tstamp = get_tstamp()
    if isinstance(tstamp, str):
        fname += f"_{tstamp}"
    fname += ".pth"
    if save_dir is None:
        save_dir = "./chkpt"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(model.state_dict(), os.path.join(save_dir, fname))
    return


def search_for_models(
    fpattern=None, directory=None, extension=".pth", recursive=True
):
    from glob import glob

    if directory is None:
        directory = "./log/"
    if fpattern is None:
        fpattern = "*"
    if recursive:
        recursive = "**"
    else:
        recursive = ""

    search_pattern = os.path.join(directory, recursive, fpattern + extension)
    found_files = glob(search_pattern, recursive=True)
    print(f"Found {len(found_files)} matches.")
    return found_files


def load_model(network: nn.Module, fpath, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = network.load_state_dict(torch.load(fpath, map_location=device))
    print(result)
    return network


def load_saved_model_by_tstamp(tstamp, in_features, device=None):
    from model import networks

    directory = os.path.join("./log", tstamp)
    model_list = search_for_models(directory=directory)
    if len(model_list) > 1:
        print("Warning: found more than one model. Loading first.")
    args_fpath = os.path.join(directory, "args.csv")
    args_df = pd.read_csv(args_fpath)
    network_name = args_df.network[0]
    network_kwargs = eval(args_df.network_kwargs[0])
    assert isinstance(
        network_kwargs, dict
    ), f"Error loading network_kwargs from {args_fpath}"
    network = networks[network_name](in_features=in_features, **network_kwargs)
    network = load_model(network, model_list[0], device=device)
    return network


def save_args(args, fpath=None):
    import csv

    if fpath is None:
        fpath = f"./log/args_{get_tstamp()}.csv"

    argdict = args.__dict__
    col_names = list(argdict.keys())
    try:
        with open(fpath, "w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=col_names)
            writer.writeheader()
            writer.writerow(argdict)
    except IOError as ioe:
        print(f"IOError: {ioe}")


def user_input_lr(lr_star, threshold=1e-5):
    if lr_star >= threshold:
        return
    if os.uname().nodename != "grimpoteuthis":
        if lr_star < threshold ** 2:
            raise RuntimeError(
                f"lr_star = {lr_star} < {threshold**2}. Stopping."
            )
        return
    response = "x"
    while response.lower() not in "yn":
        response = input(
            f"lr_star = {lr_star} < {threshold}. Continue anyway? (y/n) "
        ).lower()
        if response == "n":
            raise RuntimeError("Stopping. lr < 1e-5.")
        elif response == "y":
            break


def create_yb_to_one_hot(batch_size, classes):

    class_to_idx = {cls.item(): idx for idx, cls in enumerate(classes)}
    num_classes = len(classes)
    y_one_hot = torch.FloatTensor(batch_size, num_classes).zero_()

    def yb_to_one_hot(yb):
        indices = (
            torch.tensor([class_to_idx[cls.item()] for cls in yb])
            .long()
            .view(-1, 1)
        )
        y_one_hot.zero_()
        y_one_hot.scatter_(1, indices, 1)
        return y_one_hot

    return yb_to_one_hot
