"""
deep-generative-demixing

Uses one or two deep generative models as prior(s) for a demixing problem.

Author: Aaron Berk <aberk@math.ubc.ca>
Copyright © 2020, Aaron Berk, all rights reserved.
Created: 15 June 2020
"""
import numpy as np
import torch
from torch import optim

from data import basic_1_8_setup
from model import SimpleVAE
import train
from viz import plot_random_images


datasets, dataloaders = basic_1_8_setup()
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

model = SimpleVAE(in_features, hidden_features, latent_features)
optimizer = optim.SGD(
    model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-3
)

(
    trainer,
    evaluator,
    val_log_handler,
    val_logger,
) = train.create_autoencoder_engines(model, optimizer)
trainer = train.add_evaluation(trainer, evaluator, val_log_handler, val_loaders)


if __name__ == "__main__":

    trainer.run(train_loader, max_epochs=2)


# # deep-generative-demixing.py ends here
