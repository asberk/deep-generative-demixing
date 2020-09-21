"""
gd_numerics_two_network

These are the numerics for the generative demixing conference paper.

Author: Aaron Berk <aberk@math.ubc.ca>
Copyright © 2020, Aaron Berk, all rights reserved.
Created:  9 September 2020

"""
import inspect
from argparse import Namespace
from time import time

import numpy as np
import torch

from basic_parser import parser
from data import single_digit_setup, single_class_setup
from model import networks
import train
from train_vae import VAETrainer
import util


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
    data_class,
    image_class,
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

    dataloaders, img_shape, classes = single_class_setup(
        data_class, image_class, ravel=ravel, batch_size=batch_size
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

    args = Namespace(
        data="single_class_setup",
        data_kwargs={
            "data_class": data_class,
            "image_class": image_class,
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

    return {
        "args": args,
        "dataloaders": dataloaders,
        "network": network,
        "criterion": criterion,
        "optimizer": optimizer,
    }


def digit_setup_function(
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

    return {
        "args": args,
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
    t00 = time()
    for digit_class, trainer in trainers.items():
        t0 = time()
        trainer.train(epochs)
        t1 = time()
        duration = util.get_hms(t1 - t0)
        print("train_duration:", duration)
    print("total_train_duration:", util.get_hms(t1 - t00))


def _device_type(args):
    if torch.cuda.is_available():
        args.__dict__.setdefault("network_device", "cuda")
    else:
        args.__dict__.setdefault("network_device", "cpu")
    return args.network_device


def main(args):
    """
    Parameters
    ----------
    args : Namespace

    Returns
    -------
    args_ : dict
        dict of Namespaces where keys in the dict correspond to digit class. The
        Namespace objects house the arguments used in training the generator for
        each digit class.
    dataloaders : dict
        Dataloaders used for each digit class.
    networks : dict
        Networks used for each digit class. Possibly trained, if
        args.load_from_tstamp is True.
    criteria : dict
        Loss function used for training each generator.
    optimizers : dict
        Optimizer used for training each generator.
    """
    args.__dict__.setdefault("load_from_tstamp", False)
    args.__dict__.setdefault("eval", False)
    args.__dict__.setdefault("return_items", True)

    batch_size = {
        "train": args.train_batch_size,
        "val": args.val_batch_size,
        "test": args.val_batch_size,
    }

    device_type = _device_type(args)
    # device = torch.device(device_type)

    print(args.__dict__)
    network_kwargs, criterion_kwargs, optimizer_kwargs = split_args(args)
    args_ = {}
    dataloaders = {}
    networks = {}
    criteria = {}
    optimizers = {}
    data_classes = [args.data_class1, args.data_class2]
    image_classes = [args.image_class1, args.image_class2]
    t00 = time()
    for data_class, image_class in zip(data_classes, image_classes):
        t0 = time()
        output = setup_function(
            data_class,
            image_class,
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
        key = (data_class, image_class)
        (
            args_[key],
            dataloaders[key],
            networks[key],
            criteria[key],
            optimizers[key],
        ) = output.values()
        t1 = time()
        duration = util.get_hms(t1 - t0)
        print("Set-up duration:", duration)
    print("Total set-up duration:", util.get_hms(t1 - t00))
    __import__("pdb").set_trace()

    if args.train:
        tstamp = util.get_tstamp()
        trainers = {}
        for key in args_.keys():
            for batch in dataloaders[key]["train"]:
                img_shape = tuple(batch[0].shape[1:])
                break

            trainers[key] = VAETrainer(
                dataloaders[key],
                networks[key],
                criteria[key],
                optimizers[key],
                unflatten=img_shape,
                auto_lr=args.auto_lr,
                base_log_path=f"./log/{tstamp}",
            )
            trainers[key].save_args(args_[key])
        train_networks(trainers, args.epochs)

    if args.load_from_tstamp:
        networks = {}
        for tstamp in args.tstamps:
            networks[tstamp] = util.load_saved_model_by_tstamp(
                tstamp, device_type
            )

    if args.eval:
        print(
            "The logic for --eval is not yet completed."
            " Do not expect results."
        )
        from eval_vae import build_evaluators

        metrics = args.__dict__.get("metrics", None)
        if metrics is None:
            metrics = {}
        vae_evaluators = build_evaluators(networks, metrics)
        print("\n\nEval results:")
        for key, evaluator in vae_evaluators.items():
            for phase, loader in dataloaders[key].items():
                evaluator.run(loader)
                print(key, phase)
                print(evaluator.state.metrics)

    if args.return_items:
        return args_, dataloaders, networks, criteria, optimizers


def run(**kwargs):
    """
    Parameters
    ----------
    digit1, digit2 : int
        Default: 1, 8 resp.
    train_batch_size, val_batch_size : int
        Default: 32, 128, resp.
    network : str
        Default: "CNN_VAE",
    network_latent_features : int
        Default: 128,
    network_device : str
    criterion : str
        Default: "default",
    criterion_lamda : float
        Default: 1.0,
    optimizer : str
        Default: "SGD",
    optimizer_lr : float
        Default: 1e-5,
    optimizer_momentum : float
        Default: 0.9,
    optimizer_weight_decay : float
        Default: 1e-3,
    epochs : int
        Default: 200,
    auto_lr : bool
        Default: True,
    train : bool
        Default: True,
    eval : bool
        Whether to run code to evaluate the model on the dataloaders.
    load_from_tstamp : bool
        Whether to attempt to load a state_dict into the model before returning
        it.
    tstamps : list of str
        Must contain a list of valid tstamps to pass to
        util.load_saved_model_by_tstamp.


    Returns
    -------
    args_ : dict
        dict of Namespaces where keys in the dict correspond to digit class. The
        Namespace objects house the arguments used in training the generator for
        each digit class.
    dataloaders : dict
        Dataloaders used for each digit class.
    networks : dict
        Networks used for each digit class. Possibly trained, if
        args.load_from_tstamp is True.
    criteria : dict
        Loss function used for training each generator.
    optimizers : dict
        Optimizer used for training each generator.

    """
    from argparse import Namespace

    network_device = "cuda" if torch.cuda.is_available() else "cpu"
    args = Namespace(
        data_class1="MNISTSubset",
        data_class2="MNISTSubset",
        image_class1=1,
        image_class2=8,
        train_batch_size=32,
        val_batch_size=128,
        network="CNN_VAE",
        network_latent_features=128,
        network_device=network_device,
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

    output = main(args)
    return output


if __name__ == "__main__":
    args = parser.parse_args()

    t0 = time()
    main(args)
    t1 = time()
    print("Total duration:", util.get_hms(t1 - t0))

# # numerics_generative_demixing.py ends here
