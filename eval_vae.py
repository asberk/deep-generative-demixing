"""
eval_vae

Utilities for loading and manipulating trained generators

Author: Aaron Berk <aberk@math.ubc.ca>
Copyright © 2020, Aaron Berk, all rights reserved.
Created: 15 September 2020

Commentary:
   
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from train import create_vae_eval_step

import train
from opt_utils import create_lr_finder
from util import get_tstamp, save_args, save_model, user_input_lr
import util
import viz



    save_image_callback = create_save_image_callback(
        fig_dir, unflatten=unflatten
    )

    def _epoch_getter():
        if hasattr(trainer, "state") and hasattr(trainer.state, "__dict__"):
            return trainer.state.__dict__.get("epoch", None)

    evaluator.add_event_handler(
        Events.ITERATION_COMPLETED(once=1),
        save_image_callback,
        epoch=_epoch_getter,
    )

    logger = Logger()

    def write_log(engine, phase):
        epoch = _epoch_getter()
        loss_value = engine.state.metrics["loss"]
        print(f"Epoch {epoch} {phase} loss {loss_value:.4f}")
        logger(f"{phase}_epoch", epoch)
        for metric_name, metric_value in engine.state.metrics.items():
            logger(f"{phase}_{metric_name}", metric_value)

    @trainer.on(Events.EPOCH_COMPLETED)
    def run_evaluator(engine):
        for phase, loader in val_loaders.items():
            with evaluator.add_event_handler(
                Events.EPOCH_COMPLETED, write_log, phase=phase
            ):
                evaluator.run(loader)

    return trainer, evaluator, logger

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_step = create_vae_eval_step(network, device)


def build_evaluators(networks, metrics):
    evaluators = {}

    for (key1, loader), (key2, network) in zip(
        dataloaders.items(), networks.items()
    ):
        assert key1 == key2
        eval_step = create_vae_eval_step(
            network, device=device, non_blocking=non_blocking
        )
        evaluators[key1] = create_autoencoder_evaluator(eval_step, metrics=metrics)
    return evaluators


class VAEEvaluator:
    def __init__(self, dataloaders, model, criterion, optimizer, **kwargs):
        self.tstamp = get_tstamp()
        self.paths = {}
        self.paths["base"] = kwargs.get("base_log_path", "./log/")
        self.paths["log"] = os.path.join(self.paths["base"], self.tstamp)
        self.paths["plot"] = self.paths["log"]
        self.paths["find_lr"] = os.path.join(self.paths["plot"], "find_lr")
        self.paths["eval_img"] = os.path.join(self.paths["plot"], "eval_img")
        self.paths["chkpt"] = os.path.join(self.paths["log"], "chkpt")

        for dir_name, directory in self.paths.items():
            if not os.path.exists(directory):
                print(f"{dir_name} directory does not exist. Creating it.")
                os.makedirs(directory)

        self.train_loader = dataloaders["train"]
        self.val_loaders = {
            phase: loader
            for phase, loader in dataloaders.items()
            if phase in ["train_eval", "val"]
        }
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

        # for MNIST this should be (1, 28, 28)
        self.unflatten = kwargs.get("unflatten", None)
        self.FIND_LR = kwargs.get("auto_lr", False)

    def find_lr(self):
        if self.FIND_LR:
            self.find_lr_fpath = os.path.join(
                self.paths["find_lr"], f"find_lr_{self.tstamp}.pdf"
            )
            only_param_group = self.optimizer.param_groups[0]
            optim_fn_kwargs = {
                k: v
                for k, v in only_param_group.items()
                if k not in ["params", "lr"]
            }

            find_lr = create_lr_finder(
                self.model,
                self.criterion,
                type(self.optimizer),
                optim_fn_kwargs=optim_fn_kwargs,
            )

            lr_star = find_lr(self.train_loader, plot_fpath=self.find_lr_fpath)
            self.lr_ = lr_star
            self.optimizer.param_groups[0]["lr"] = lr_star

    def prepare_train(self):
        self.trainer, self.evaluator, self.logger = train.create_vae_engines(
            self.model,
            self.optimizer,
            self.val_loaders,
            fig_dir=self.paths["eval_img"],
            unflatten=self.unflatten,
        )

    def train(self, max_epochs):
        """Trains a model on data.
        Parameters
        ----------
        max_epochs: int
            maximum number of epochs to train for
        """
        self.find_lr()
        self.prepare_train()
        self.trainer.run(self.train_loader, max_epochs=max_epochs)
        self.logger.save(os.path.join(self.paths["log"], "val_log.csv"))
        save_model(
            self.model,
            self.model._get_name(),
            epoch=max_epochs,
            score_name="val_loss",
            score_value=self.logger.log["val_loss"][-1],
            tstamp=self.tstamp,
            save_dir=self.paths["chkpt"],
        )


parent_dir = "./log/"
tstamps = sorted(["20200914-105758-237836", "20200914-105759-069972"])

tstamp = "20200914-105758-237836"


for tstamp in tstamps:
    directory = util.search_for_tstamp_directory(tstamp, parent_dir)
    val_log = util.load_val_log(directory)
    fig, ax = viz.plot_val_log(val_log, title=tstamp)
    plt.show()


util.load_saved_model_by_tstamp(tstamp, (1, 784), device="cpu")


# # eval_vae.py ends here
