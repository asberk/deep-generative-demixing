import os
from argparse import Namespace
import numpy as np
import matplotlib.pyplot as plt


import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.utils import make_grid

import data as _data
from model import CNN_VAE
from util import save_args, get_tstamp

args = Namespace(
    image_class=3,
    transform_type="basic",
    train_batch_size=32,
    val_batch_size=128,
    latent_features=128,
    optim_fn="Adam",
    optim_fn_kwargs={"lr": 1e-3, "weight_decay": 1e-3},
    num_epochs=30,
)


def get_load_transform():
    return transforms.Compose(
        [
            transforms.RandomCrop((28, 28)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )


transform_options = {
    "None": transforms.ToTensor(),
    "basic": get_load_transform(),
}

batch_size = {
    "train": args.train_batch_size,
    "train_eval": args.val_batch_size,
    "val": args.val_batch_size,
    "test": args.val_batch_size,
}


def interleave(A, B):
    assert A.ndim == 4
    assert B.ndim == 4
    assert all(a == b for a, b in zip(A.shape, B.shape))
    C = torch.stack((A, B)).transpose(0, 1).reshape(-1, *A.shape[1:])
    return C


def plot_batch(batch, fpath=None):
    if not isinstance(batch, torch.Tensor):
        images, targets = batch
    else:
        assert batch.ndim == 4
        images = batch
    X = make_grid(images).numpy().transpose((1, 2, 0))
    # X = X - X.min()
    # X = X / X.max()
    plt.imshow(X)
    plt.axis("off")
    plt.tight_layout()
    if fpath is None:
        plt.show()
    else:
        plt.savefig(fname=fpath, dpi=300, bbox_inches="tight")
        plt.close("all")


def load_data():
    return _data.all_class_setup(
        "CIFAR10Subset",
        batch_size=batch_size,
        on_load_transform=get_load_transform(),
    )
    # return _data.single_class_setup(
    #     "FMNISTSubset",
    #     args.image_class,
    #     batch_size=batch_size,
    #     on_load_transform=get_load_transform(),
    # )


class ToyVAE(nn.Module):
    def __init__(self, in_channels, latent_features, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(device, str):
            device = torch.device(device)

        super().__init__()

        self._device = device

        # Try kernel_size=4, stride=2, padding=1
        self.encoder = nn.Sequential(
            self._conv(in_channels, latent_features // 4),
            self._conv(latent_features // 4, latent_features // 2),
            self._conv(latent_features // 2, latent_features),
        )
        self.decoder = nn.Sequential(
            self._deconv(
                latent_features, latent_features // 2, kernel_size=4, dilation=2
            ),
            self._deconv(
                latent_features // 2,
                latent_features // 4,
                kernel_size=4,
                dilation=1,
                padding=1,
            ),
            self._deconv(
                latent_features // 4,
                in_channels,
                kernel_size=3,
                dilation=1,
                padding=1,
                relu=False,
                output_padding=1,
            ),
        )
        self.q_mean = self._linear(latent_features, latent_features)
        self.q_logvar = self._linear(latent_features, latent_features)

    def _conv(self, in_channels, out_channels, kernel_size=(3, 3)):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )

    def _deconv(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        dilation=1,
        stride=2,
        padding=0,
        output_padding=0,
        relu=True,
    ):
        activn = nn.ReLU() if relu else nn.Sigmoid()
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
            ),
            nn.BatchNorm2d(out_channels),
            activn,
        )

    def _linear(self, in_features, out_features, relu=True):
        if relu:
            return nn.Sequential(
                nn.Linear(in_features, out_features), nn.ReLU(),
            )
        return nn.Linear(in_features, out_features)

    def _q(self, encoded):
        unrolled = encoded.view(encoded.size(0), -1)
        return self.q_mean(unrolled), self.q_logvar(unrolled)

    def reparametrize(self, mu, log_var):
        eps = torch.randn_like(mu).to(self._device)
        return mu + eps * log_var.mul_(0.5).exp_()

    def forward(self, input):
        encoded = self.encoder(input)
        mu, log_var = self._q(encoded)
        z = self.reparametrize(mu, log_var)
        z.unsqueeze_(2).unsqueeze_(2)
        output = self.decoder(z)
        return output, mu, log_var


def prepare_network():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # network = CNN_VAE(3, args.latent_features, device=device)
    network = ToyVAE(3, args.latent_features, device=device)
    return network, device


# def load_data():
#     return _data.single_class_setup(
#         "MNISTSubset",
#         args.image_class,
#         batch_size=batch_size,
#         on_load_transform=get_load_transform(),
#     )


# def prepare_network():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     network = CNN_VAE(1, args.latent_features, device=device)
#     return network, device


def AutoEncoderLoss(lamda=None):
    if lamda is None:
        lamda = 1.0
    if not isinstance(lamda, torch.Tensor):
        lamda = torch.tensor(lamda).float()

    bce_criterion = nn.BCELoss(reduction="sum")

    def loss_fn(x_recon, x_true, mu, log_var):
        BCE = bce_criterion(x_recon, x_true)
        # KLD = mu.pow(2) + log_var.exp() - log_var - 1
        # KLD = KLD.mul_(0.5).sum()
        # total_loss = BCE + lamda.to(KLD.device) * KLD
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        total_loss = BCE + KLD
        return total_loss

    return loss_fn


def create_train_step(network, criterion, optimizer, device=None):
    if device is None:
        device = network._device
    if isinstance(device, str):
        device = torch.device(device)

    def train_step(batch):
        network.train()
        optimizer.zero_grad()
        x_true, targets = batch
        x_true = x_true.to(device)
        with torch.set_grad_enabled(True):
            x_recon, mu, log_var = network(x_true)
            loss = criterion(x_recon, x_true, mu, log_var)
            loss.backward()
            optimizer.step()
        return loss.item()

    return train_step


def create_eval_step(network, criterion, device=None, output_folder=None):
    if device is None:
        device = network._device
    if isinstance(device, str):
        device = torch.device(device)

    mse_func = nn.MSELoss()

    if output_folder is None:
        output_folder = f"val_images_{get_tstamp()}"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    def eval_step(batch, epoch_number=None):
        network.eval()
        x_true, targets = batch
        x_true = x_true.to(device)
        with torch.set_grad_enabled(False):
            x_recon, mu, log_var = network(x_true)
            loss = criterion(x_recon, x_true, mu, log_var)
            mse_val = mse_func(x_recon, x_true)
        if epoch_number is None:
            tag = get_tstamp()
        else:
            tag = f"{epoch_number:03d}"

        xt_ = x_true.cpu().detach()
        xr_ = x_recon.cpu().detach()
        interleaved = interleave(xt_, xr_)
        plot_batch(
            interleaved, os.path.join(output_folder, f"val_{tag}.pdf"),
        )
        return loss.item(), mse_val.item()

    return eval_step


def create_trainer(
    network, criterion, optim_fn=None, optim_fn_kwargs=None, device=None
):

    if device is None:
        device = network._device
    if isinstance(device, str):
        device = torch.device(device)

    if optim_fn_kwargs is None:
        optim_fn_kwargs = {}
    optim_fn_kwargs.setdefault("lr", 1e-3)
    # optim_fn_kwargs.setdefault("momentum", 0.9)
    optim_fn_kwargs.setdefault("weight_decay", 1e-5)

    if optim_fn is None:
        optim_fn = optim.Adam

    optimizer = optim_fn(network.parameters(), **optim_fn_kwargs)

    annealer = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    train_step = create_train_step(network, criterion, optimizer, device)
    eval_step = create_eval_step(network, criterion, device)

    def _print_train_info(train_info, epoch):
        print(
            "train epoch",
            train_info["epoch"][-1],
            "loss",
            np.mean(
                [
                    ell
                    for ell, ep in zip(train_info["loss"], train_info["epoch"])
                    if ep == epoch
                ]
            ),
        )

    def _print_val_info(val_info, epoch):
        print(
            "val epoch",
            val_info["epoch"][-1],
            "loss",
            np.mean(
                [
                    ell
                    for ell, ep in zip(val_info["loss"], val_info["epoch"])
                    if ep == epoch
                ]
            ),
            "mse",
            np.mean(
                [
                    em
                    for em, ep in zip(val_info["mse"], val_info["epoch"])
                    if ep == epoch
                ]
            ),
        )

    def trainer(train_loader, val_loader, num_epochs):
        train_info = {"epoch": [], "batch": [], "loss": []}
        val_info = {"epoch": [], "batch": [], "loss": [], "mse": []}
        for epoch in range(num_epochs):

            # Train phase
            for i, batch in enumerate(train_loader):
                train_batch_loss_value = train_step(batch)
                train_info["epoch"].append(epoch)
                train_info["batch"].append(i)
                train_info["loss"].append(train_batch_loss_value)

            _print_train_info(train_info, epoch)

            for i, batch in enumerate(val_loader):
                val_batch_loss_value, val_batch_mse_value = eval_step(
                    batch, epoch
                )
                val_info["epoch"].append(epoch)
                val_info["batch"].append(i)
                val_info["loss"].append(val_batch_loss_value)
                val_info["mse"].append(val_batch_mse_value)
                break

            _print_val_info(val_info, epoch)

            annealer.step()

        return network, train_info, val_info

    return trainer


optimizers = {"SGD": optim.SGD, "Adam": optim.Adam}


if __name__ == "__main__":
    tstamp = get_tstamp()
    dataloaders, img_shape, classes = load_data()

    # for batch in dataloaders["train"]:
    #     images, targets = batch
    #     break

    # plot_batch(images)

    network, device = prepare_network()
    criterion = AutoEncoderLoss()
    optim_fn = optimizers[args.optim_fn]
    trainer = create_trainer(
        network,
        criterion,
        optim_fn=optim_fn,
        optim_fn_kwargs=args.optim_fn_kwargs,
        device=device,
    )
    final_network, train_info, val_info = trainer(
        dataloaders["train"], dataloaders["val"], args.num_epochs
    )
    torch.save(final_network.cpu().state_dict(), f"network_{tstamp}.pth")
    save_args(args, fpath=f"args_{tstamp}.csv")
