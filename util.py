import os
from datetime import datetime
from collections import defaultdict
import numpy as np
import pandas as pd
import torch


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
