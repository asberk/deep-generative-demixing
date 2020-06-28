"""
train_cnn_cvae_script

Trains a convolutional conditional variational autoencoder on the Fashion-MNIST dataset.

Author: Aaron Berk <aberk@math.ubc.ca>
Copyright © 2020, Aaron Berk, all rights reserved.
Created: 28 June 2020

Commentary:
    You want to train this on Google Colab or something of that ilk.
"""
import os
import sys
from collections import defaultdict
from argparse import Namespace

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from torchvision.utils import save_image

import skorch
from skorch.helper import predefined_split

from torchsummary import summary

from data import load_data_fns
from model import networks
from train import criteria, create_cvae_engines, optimizers
from util import get_tstamp, save_args, save_model

args = Namespace(
    data="basic_fmnist_setup",
    data_kwargs={"ravel": False, "batch_size": 128},
    network="CCVAE",
    network_kwargs={"latent_features": 64},
    criterion="default",
    criterion_kwargs={"lamda": 1.0},
    optim_fn="Adam",
    optim_fn_kwargs={"lr": 1e-3, "weight_decay": 1e-3},
    max_epochs=200,
)
tstamp = get_tstamp()
log_path = f"./log/{tstamp}/"
plot_path = log_path
eval_img_path = os.path.join(plot_path, "eval_img")
chkpt_path = os.path.join(log_path, "chkpt")

dataloaders, in_channels, num_classes = load_data_fns[args.data](
    **args.data_kwargs
)
train_loader = dataloaders["train"]
val_loaders = {
    phase: loader
    for phase, loader in dataloaders.items()
    if phase in ["train", "val"]
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
network = networks[args.network](
    in_channels=in_channels,
    num_classes=num_classes,
    device=device,
    **args.network_kwargs,
)
network = network.to(device)
criterion = criteria[args.criterion](**args.criterion_kwargs)
optimizer = optimizers[args.optim_fn](
    network.parameters(), **args.optim_fn_kwargs
)

trainer, evaluator, logger = create_cvae_engines(
    network, criterion, optimizer, val_loaders, device, fig_dir=eval_img_path
)


if __name__ == "__main__":
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    if not os.path.exists(chkpt_path):
        os.makedirs(chkpt_path)

    trainer.run(dataloaders["train"], max_epochs=args.max_epochs)

    if args.data_kwargs.get("batch_size", None) is None:
        args.data_kwargs["batch_size"] = dataloaders["train"].batch_size
    logger.save(os.path.join(log_path, "val_log.csv"))
    save_args(args, os.path.join(log_path, "args.csv"))

    save_model(
        network,
        network._get_name(),
        epoch=args.max_epochs,
        score_name="val_loss",
        score_value=logger.log["val_loss"][-1],
        tstamp=tstamp,
        save_dir=chkpt_path,
    )


# # train_cnn_cvae_script.py ends here
