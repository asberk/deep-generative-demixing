"""
gd_numerics_two_network

These are the numerics for the generative demixing conference paper.

Author: Aaron Berk <aberk@math.ubc.ca>
Copyright © 2020, Aaron Berk, all rights reserved.
Created:  9 September 2020

"""
import os
import inspect
from argparse import Namespace
import numpy as np
import torch

from basic_parser import parser
from data import single_digit_setup
from model import networks
import train
from train_vae import VAETrainer
from util import get_tstamp, save_args


tstamp = get_tstamp()


def _key_helper(key):
    return "_".join(key.split("_")[1:])


def split_args(args):
    network_kwargs = {
        _key_helper(k): v
        for k, v in args.__dict__.items()
        if ("network_" in k) and (v is not None)
    }
    criterion_kwargs = {
        _key_helper(k): v
        for k, v in args.__dict__.items()
        if ("criterion_" in k) and (v is not None)
    }
    optimizer_kwargs = {
        _key_helper(k): v
        for k, v in args.__dict__.items()
        if ("optimizer_" in k) and (v is not None)
    }
    return network_kwargs, criterion_kwargs, optimizer_kwargs


def setup_function(
    digit_class,
    max_epochs,
    batch_size,
    auto_lr,
    network_name,
    network_kwargs,
    criterion_name,
    criterion_kwargs,
    optimizer_name,
    optimizer_kwargs,
):

    Network = networks[network_name]
    network_arg_names = inspect.getfullargspec(Network).args
    network_arg_names = [x for x in network_arg_names if x != "self"]

    if "in_channels" in network_arg_names:
        ravel = False
    else:
        ravel = True

    dataloaders, img_shape, classes = single_digit_setup(
        digit_class, ravel=ravel, batch_size=batch_size
    )

    if "in_channels" in network_arg_names:
        emsg = f"Expected images, not vectors; got img_shape = {img_shape}"
        assert len(img_shape) >= 3, emsg
        in_channels = img_shape[-3]
        network_kwargs["in_channels"] = in_channels
    elif "in_features" in network_arg_names:
        in_features = np.prod(img_shape)
        network_kwargs["in_features"] = in_features

    for key in network_kwargs.keys():
        if key not in network_arg_names:
            emsg = (
                f"Unexpected argument for Network {network_name}. "
                f"Valid args are:\n  {network_arg_names}"
            )
            raise ValueError(emsg)

    criterion_kwargs.setdefault("device", network_kwargs.get("device", "cpu"))

    print(network_kwargs)
    network = Network(**network_kwargs)
    criterion = train.criteria[criterion_name](**criterion_kwargs)
    optim_fn = train.optimizers[optimizer_name]
    optimizer = optim_fn(network.parameters(), **optimizer_kwargs)
    vae_trainer = VAETrainer(
        dataloaders,
        network,
        criterion,
        optimizer,
        unflatten=(1, 28, 28),
        auto_lr=auto_lr,
        base_log_path=f"./log/{tstamp}",
    )

    args = Namespace(
        data="single_digit_setup",
        data_kwargs={
            "digit_class": digit_class,
            "ravel": ravel,
            "batch_size": batch_size,
        },
        max_epochs=max_epochs,
        network=network_name,
        network_kwargs=network_kwargs,
        criterion=criterion_name,
        criterion_kwargs=criterion_kwargs,
        optim_fn=optimizer_name,
        optim_fn_kwargs=optimizer_kwargs,
    )

    save_args(args, os.path.join(vae_trainer.paths["log"], "args.csv"))

    return {
        "trainer": vae_trainer,
        "dataloaders": dataloaders,
        "network": network,
        "criterion": criterion,
        "optimizer": optimizer,
    }


def train_networks(trainers, epochs):
    """trains the VAETrainer objects in the dict `trainers` for `epochs` epochs.

    Parameters
    ----------
    trainers: dict of VAETrainer
        having keys equal to digit_classes
    epochs: int
        number of epochs for which to train each model.

    """
    for digit_class, trainer in trainers.items():
        trainer.train(epochs)


def _device_type(args):
    if torch.cuda.is_available():
        args.__dict__.setdefault("network_device", "cuda")
    else:
        args.__dict__.setdefault("network_device", "cpu")
    return args.network_device


def main(args):
    batch_size = {
        "train": args.train_batch_size,
        "val": args.val_batch_size,
        "test": args.val_batch_size,
    }

    _ = _device_type(args)
    # device = torch.device(device_type)

    print(args.__dict__)
    network_kwargs, criterion_kwargs, optimizer_kwargs = split_args(args)
    objects = {}
    digit_classes = [args.digit1, args.digit2]
    for digit_class in digit_classes:
        objects[digit_class] = setup_function(
            digit_class,
            args.epochs,
            batch_size,
            args.auto_lr,
            args.network,
            network_kwargs,
            args.criterion,
            criterion_kwargs,
            args.optimizer,
            optimizer_kwargs,
        )

    if args.train:
        trainers = {dc: objects[dc]["trainer"] for dc in digit_classes}
        train_networks(trainers, args.epochs)


def run(**kwargs):
    from argparse import Namespace

    args = Namespace(
        digit1=1,
        digit2=8,
        train_batch_size=32,
        val_batch_size=128,
        network="CNN_VAE",
        network_latent_features=128,
        network_device="cuda",
        criterion="default",
        criterion_lamda=1.0,
        optimizer="SGD",
        optimizer_lr=1e-5,
        optimizer_momentum=0.9,
        optimizer_weight_decay=1e-3,
        epochs=200,
        auto_lr=True,
        train=True,
    )

    for key, value in kwargs.items():
        args.__dict__[key] = value

    main(args)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

# # numerics_generative_demixing.py ends here
