"""
train_vae

Class for training a VAE on a digit.

Author: Aaron Berk <aberk@math.ubc.ca>
Copyright © 2020, Aaron Berk, all rights reserved.
Created:  9 September 2020
"""
import os

import train
from opt_utils import create_lr_finder
from util import get_tstamp, save_args, save_model, user_input_lr


class VAETrainer:
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


# # train_vae.py ends here
