"""
SCRATCH

Some utilities for testing the gd_numerics code

Author: Aaron Berk <aberk@math.ubc.ca>
Copyright © 2020, Aaron Berk, all rights reserved.
Created: 21 September 2020

Commentary:
   
"""
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import hiddenlayer as hl

from gd_numerics_two_network import run


def load_no_train_cifar_mnist():
    """
    Returns
    -------
    output : tuple
        args : Namespace
        dataloaders: dict of dict of DataLoaders
        networks : dict of nn.Module
        criteria : dict of criteria
        optimizers : dict of optim.Optimizer
    """
    return run(train=False, data_class1="CIFAR10Subset", image_class1=3)


def load_train_cifar_mnist():
    return run(
        train=True,
        data_class1="CIFAR10Subset",
        image_class1=3,
        auto_lr=False,
        optimizer_lr=1e-5,
    )


def get_a_batch(dataloaders):
    if isinstance(dataloaders, DataLoader):
        for batch in dataloaders:
            return batch
    dataloader = list(dataloaders.values())[0]
    return get_a_batch(dataloader)


class Encoder_20200914(nn.Module):
    def __init__(self, in_channels, latent_features, device):
        """
        CNN variant of a variational autoencoder.

        encode
        ---------------------------
        torch.Size([1,   1, 28, 28]) ⤸
        torch.Size([1,  16, 13, 13]) ⤸
        torch.Size([1,  32,  5,  5]) ⤸
        torch.Size([1, 128,  1,  1]) ⤸
        torch.Size([1, 128,  1,  1]) ⤸
        (
            torch.Size([1, latent_features]),
            torch.Size([1, latent_features])
        )

        decode
        ------------------------------
        torch.Size([1, latent_features])     ⤸
        torch.Size([1, 32,  2,  2])          ⤸
        torch.Size([1, 32,  8,  8])          ⤸
        torch.Size([1, 32, 14, 14])          ⤸
        torch.Size([1, 16, 20, 20])          ⤸
        torch.Size([1, 16, 26, 26])          ⤸
        torch.Size([1, in_channels, 28, 28])
        """
        super().__init__()

        self._device = device if torch.cuda.is_available() else "cpu"
        self.conv1 = nn.Conv2d(in_channels, 16, (3, 3))
        self.conv2 = nn.Conv2d(16, 32, (3, 3))
        self.conv3 = nn.Conv2d(32, 2 * latent_features, (3, 3))
        self.lin1 = nn.Linear(2 * latent_features, latent_features)
        self.lin2 = nn.Linear(2 * latent_features, latent_features)

    def forward(self, input):
        print("Encoding")
        print("input_shape:", input.shape)
        X = torch.max_pool2d(torch.relu(self.conv1(input)), (2, 2))
        print(X.shape)
        X = torch.max_pool2d(torch.relu(self.conv2(X)), (2, 2))
        print(X.shape)
        X = torch.max_pool2d(torch.relu(self.conv3(X)), (2, 2))
        print(X.shape)
        X = nn.functional.adaptive_avg_pool2d(X, (1, 1))
        print(X.shape)
        X = X.view(X.size(0), -1)
        print(X.shape)
        mu = self.lin1(X)
        log_var = self.lin2(X)
        print("mu:", mu.shape)
        print("log_var:", log_var.shape)
        return mu, log_var


class Decoder_20200914(nn.Module):
    def __init__(self, in_channels, latent_features, device):
        """
        CNN variant of a variational autoencoder.

        encode
        ---------------------------
        torch.Size([1,   1, 28, 28]) ⤸
        torch.Size([1,  16, 13, 13]) ⤸
        torch.Size([1,  32,  5,  5]) ⤸
        torch.Size([1, 128,  1,  1]) ⤸
        torch.Size([1, 128,  1,  1]) ⤸
        (
            torch.Size([1, latent_features]),
            torch.Size([1, latent_features])
        )

        decode
        ------------------------------
        torch.Size([1, latent_features])     ⤸
        torch.Size([1, 32,  2,  2])          ⤸
        torch.Size([1, 32,  8,  8])          ⤸
        torch.Size([1, 32, 14, 14])          ⤸
        torch.Size([1, 16, 20, 20])          ⤸
        torch.Size([1, 16, 26, 26])          ⤸
        torch.Size([1, in_channels, 28, 28])
        """
        super().__init__()

        self.lin3 = nn.Linear(latent_features, 32 * 2 * 2)
        self.convt1 = nn.ConvTranspose2d(32, 32, (4, 4), dilation=2, stride=1)
        self.convt2 = nn.ConvTranspose2d(32, 32, (4, 4), dilation=2, stride=1)
        self.convt3 = nn.ConvTranspose2d(32, 16, (4, 4), dilation=2, stride=1)
        self.convt4 = nn.ConvTranspose2d(16, 16, (4, 4), dilation=2, stride=1)
        self.convt5 = nn.ConvTranspose2d(
            16, in_channels, (3, 3), dilation=1, stride=1
        )

    def forward(self, z):
        print("Decoding")
        print("input_shape:", z.shape)
        X = self.lin3(z).view(-1, 32, 2, 2)
        print(X.shape)
        X = torch.relu(self.convt1(X))
        print(X.shape)
        X = torch.relu(self.convt2(X))
        print(X.shape)
        X = torch.relu(self.convt3(X))
        print(X.shape)
        X = torch.relu(self.convt4(X))
        print(X.shape)
        X = torch.sigmoid(self.convt5(X))
        print("img_shape:", X.shape)
        return X


class CNN_VAE(nn.Module):
    def __init__(self, in_channels, latent_features, device):
        """
        CNN variant of a variational autoencoder.

        encode
        ---------------------------
        torch.Size([1,   1, 28, 28]) ⤸
        torch.Size([1,  16, 13, 13]) ⤸
        torch.Size([1,  32,  5,  5]) ⤸
        torch.Size([1, 128,  1,  1]) ⤸
        torch.Size([1, 128,  1,  1]) ⤸
        (
            torch.Size([1, latent_features]),
            torch.Size([1, latent_features])
        )

        decode
        ------------------------------
        torch.Size([1, latent_features])     ⤸
        torch.Size([1, 32,  2,  2])          ⤸
        torch.Size([1, 32,  8,  8])          ⤸
        torch.Size([1, 32, 14, 14])          ⤸
        torch.Size([1, 16, 20, 20])          ⤸
        torch.Size([1, 16, 26, 26])          ⤸
        torch.Size([1, in_channels, 28, 28])
        """
        super().__init__()

        self._device = device if torch.cuda.is_available() else "cpu"

        self.encoder = Encoder_20200914(in_channels, latent_features, device)
        self.decoder = Decoder_20200914(in_channels, latent_features, device)

    def reparametrize(self, mu, log_var):
        eps = torch.randn_like(mu).to(self._device)
        return mu + eps * log_var.mul_(0.5).exp_()

    def forward(self, input):
        mu, log_var = self.encoder(input)
        z = self.reparametrize(mu, log_var)
        output = self.decoder(z)
        return output, mu, log_var


if __name__ == "__main__":

    # output = load_no_train_cifar_mnist()
    # output = load_train_cifar_mnist()
    # batch = get_a_batch(output[1])

    the_encoder = Encoder_20200914(1, 128, "cpu")
    the_decoder = Decoder_20200914(1, 128, "cpu")
    the_autoencoder = CNN_VAE(1, 128, "cpu")

    the_autoencoder(torch.zeros((1, 1, 28, 28)))

    # encoder_graph = hl.build_graph(
    #     the_encoder,
    #     torch.zeros((1, 1, 28, 28)),
    #     transforms=hl.transforms.SIMPLICITY_TRANSFORMS,
    # )
    # decoder_graph = hl.build_graph(
    #     the_decoder,
    #     torch.zeros((1, 128)),
    #     transforms=hl.transforms.SIMPLICITY_TRANSFORMS,
    # )
    # autoencoder_graph = hl.build_graph(
    #     the_autoencoder,
    #     torch.zeros((1, 1, 28, 28)),
    #     transforms=hl.transforms.SIMPLICITY_TRANSFORMS,
    # )
    # encoder_graph.theme = hl.graph.THEMES[
    #     "blue"
    # ].copy()  # Two options: basic and blue
    # decoder_graph.theme = hl.graph.THEMES[
    #     "blue"
    # ].copy()  # Two options: basic and blue
    # autoencoder_graph.theme = hl.graph.THEMES[
    #     "blue"
    # ].copy()  # Two options: basic and blue

    # encoder_graph.save(path="./log/encoder_graph_20200914.pdf")
    # decoder_graph.save(path="./log/decoder_graph_20200914.pdf")
    # autoencoder_graph.save(path="./log/autoencoder_graph_20200914.pdf")


# # scratch.py ends here
