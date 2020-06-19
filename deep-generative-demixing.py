"""
deep-generative-demixing

Uses one or two deep generative models as prior(s) for a demixing problem.

Author: Aaron Berk <aberk@math.ubc.ca>
Copyright © 2020, Aaron Berk, all rights reserved.
Created: 15 June 2020
"""
import os
from argparse import Namespace
import numpy as np

from data import load_data_fns
from model import networks
from opt_utils import create_lr_finder
import train
from util import get_tstamp, save_args, user_input_lr

# from viz import plot_random_images

args = Namespace(
    data="basic_1_8_setup",
    data_kwargs={"ravel": True, "batch_size": 128},
    network="SimpleVAE",
    network_kwargs={
        "hidden_features": 256,
        # "num_layers": 4,
        "latent_features": 16,
        "dropout_probability": 0.3,
    },
    criterion="default",
    criterion_kwargs={"lamda": 1.0},
    optim_fn="SGD",
    optim_fn_kwargs={"lr": None, "momentum": 0.9, "weight_decay": 1e-3},
    max_epochs=200,
)

datasets, dataloaders = load_data_fns[args.data]()
# plot_random_images(dset_18_test, k=32, nr=4)

val_loaders = {
    phase: loader
    for phase, loader in dataloaders.items()
    if phase in ["train_eval", "val"]
}
train_loader = dataloaders["train"]

img_shape = datasets["train"][0][0].size()
in_features = np.prod(img_shape)
hidden_features = 5
latent_features = 3

model = networks[args.network](in_features, **args.network_kwargs)
criterion = train.criteria[args.criterion](**args.criterion_kwargs)
optim_fn = train.optimizers[args.optim_fn]
optim_fn_kwargs = args.optim_fn_kwargs

find_lr = create_lr_finder(
    model, criterion, optim_fn, optim_fn_kwargs=optim_fn_kwargs,
)

tstamp = get_tstamp()
log_path = f"./log/{tstamp}/"
plot_path = log_path
find_lr_path = os.path.join(plot_path, "find_lr")
find_lr_fpath = os.path.join(find_lr_path, f"find_lr_{tstamp}.pdf")
eval_img_path = os.path.join(plot_path, "eval_img")


if __name__ == "__main__":

    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    if not os.path.exists(find_lr_path):
        os.makedirs(find_lr_path)

    if optim_fn_kwargs.get("lr", None) is None:
        lr_star = find_lr(train_loader, plot_fpath=find_lr_fpath)
        print("lr_star", lr_star)
        args.optim_fn_kwargs["lr"] = lr_star
        optim_fn_kwargs["lr"] = lr_star
        user_input_lr(lr_star)

    optimizer = optim_fn(model.parameters(), **optim_fn_kwargs)

    (
        trainer,
        evaluator,
        val_log_handler,
        val_logger,
    ) = train.create_autoencoder_engines(
        model, optimizer, fig_dir=eval_img_path, unflatten=(1, 28, 28)
    )
    trainer = train.add_evaluation(
        trainer, evaluator, val_log_handler, val_loaders
    )

    trainer.run(train_loader, max_epochs=args.max_epochs)

    if args.data_kwargs.get("batch_size", None) is None:
        args.data_kwargs["batch_size"] = dataloaders["train"].batch_size
    val_logger.save(os.path.join(log_path, "val_log.csv"))
    save_args(args, os.path.join(log_path, "args.csv"))

# # deep-generative-demixing.py ends here
