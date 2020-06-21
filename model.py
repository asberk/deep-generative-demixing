"""
model

Author: Aaron Berk <aberk@math.ubc.ca>
Copyright © 2020, Aaron Berk, all rights reserved.
Created: 15 June 2020

"""
import torch
from torch import nn


class SimpleVAE(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        latent_features,
        dropout_probability=0.4,
        device=None,
    ):
        """
        VariationalAutoEncoder(in_features, hidden_features)
        
        Parameters
        ----------
        in_features : int
            Input dimension of the network
        hidden_features : int
            Number of features in the hidden layer
        latent_features : int
            Size of the latent/encoded representation
        """
        super().__init__()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self._args = (
            in_features,
            hidden_features,
            latent_features,
            dropout_probability,
            device,
        )
        self.in_features = in_features
        self.dropout_probability = dropout_probability
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc21 = nn.Linear(hidden_features, latent_features)
        self.fc22 = nn.Linear(hidden_features, latent_features)
        self.fc3 = nn.Linear(latent_features, hidden_features)
        self.fc4 = nn.Linear(hidden_features, in_features)

    def encode(self, input):
        dropout = lambda x: nn.functional.dropout(
            x, self.dropout_probability, self.training
        )

        h1 = dropout(torch.relu(self.fc1(input)))
        mu, log_var = self.fc21(h1), self.fc22(h1)
        return mu, log_var

    def reparametrize(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        dropout = lambda x: nn.functional.dropout(
            x, self.dropout_probability, self.training
        )
        h3 = dropout(torch.relu(self.fc3(z)))
        out = torch.sigmoid(self.fc4(h3))
        return out

    def forward(self, input):
        mu, log_var = self.encode(input.view(-1, self.in_features))
        z = self.reparametrize(mu, log_var)
        out = self.decode(z)
        return out, mu, log_var


class SimpleConditionalVAE(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        latent_features,
        num_classes,
        dropout_probability=0.4,
        device=None,
    ):
        """
        SimpleConditionalVAE(
            in_features,
            hidden_features,
            latent_features,
            num_classes,
            dropout_probability=0.4,
            device=None,
        )
        
        Parameters
        ----------
        in_features : int
            Input dimension of the network
        hidden_features : int
            Number of features in the hidden layer
        latent_features : int
            Size of the latent/encoded representation
        num_classes : int
        dropout_probability : float
            Default: 0.4
        device : torch.device
        """
        super().__init__()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self._args = (
            in_features,
            hidden_features,
            latent_features,
            dropout_probability,
            device,
        )
        self.in_features = in_features
        self.num_classes = num_classes
        self.dropout_probability = dropout_probability
        self.fc1 = nn.Linear(int(in_features + num_classes), hidden_features)
        self.fc21 = nn.Linear(hidden_features, latent_features)
        self.fc22 = nn.Linear(hidden_features, latent_features)
        self.fc3 = nn.Linear(
            int(latent_features + num_classes), hidden_features
        )
        self.fc4 = nn.Linear(hidden_features, in_features)

    def encode(self, input):
        def dropout(x):
            return nn.functional.dropout(
                x, self.dropout_probability, self.training
            )

        h1 = dropout(torch.relu(self.fc1(input)))
        mu, log_var = self.fc21(h1), self.fc22(h1)
        return mu, log_var

    def reparametrize(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        def dropout(x):
            return nn.functional.dropout(
                x, self.dropout_probability, self.training
            )

        h3 = dropout(torch.relu(self.fc3(z)))
        out = torch.sigmoid(self.fc4(h3))
        return out

    def forward(self, input):
        Xb, y_oh = input
        Xb = Xb.view(-1, self.in_features)
        enc_input = torch.cat((Xb, y_oh[: Xb.size(0), :]), dim=1)
        mu, log_var = self.encode(enc_input)
        z = self.reparametrize(mu, log_var)
        dec_input = torch.cat((z, y_oh[: Xb.size(0), :]), dim=1)
        out = self.decode(dec_input)
        return out, mu, log_var


class FullyConnectedVAE(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        num_layers,
        latent_features,
        dropout_probability=0.4,
        device=None,
    ):
        """
        FullyConnectedVAE(
            in_features,
            hidden_features,
            num_layers,
            latent_features,
            dropout_probability=0.4,
            device=None,
        )
        """
        super().__init__()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self._args = (
            in_features,
            hidden_features,
            num_layers,
            latent_features,
            dropout_probability,
            device,
        )
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.num_layers = num_layers
        self.latent_features = latent_features
        self.dropout_probability = dropout_probability
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc21 = nn.Linear(hidden_features, latent_features)
        self.fc22 = nn.Linear(hidden_features, latent_features)
        self.fc3 = nn.Linear(latent_features, hidden_features)
        self.fc4 = nn.Linear(hidden_features, in_features)

        if num_layers > 2:
            self.enc_layers = nn.ModuleList(
                [
                    nn.Linear(hidden_features, hidden_features)
                    for _ in range(num_layers - 2)
                ]
            )
            self.dec_layers = nn.ModuleList(
                [
                    nn.Linear(hidden_features, hidden_features)
                    for _ in range(num_layers - 2)
                ]
            )

    def encode(self, input):
        dropout = lambda x: nn.functional.dropout(
            x, self.dropout_probability, self.training
        )

        h1 = dropout(torch.relu(self.fc1(input)))
        if self.num_layers > 2:
            for layer in self.enc_layers:
                h1 = dropout(torch.relu(layer(h1)))
        mu, log_var = self.fc21(h1), self.fc22(h1)
        return mu, log_var

    def reparametrize(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        dropout = lambda x: nn.functional.dropout(
            x, self.dropout_probability, self.training
        )
        h3 = dropout(torch.relu(self.fc3(z)))
        if self.num_layers > 2:
            for layer in self.dec_layers:
                h3 = dropout(torch.relu(layer(h3)))
        out = torch.sigmoid(self.fc4(h3))
        return out

    def forward(self, input):
        mu, log_var = self.encode(input.view(-1, self.in_features))
        z = self.reparametrize(mu, log_var)
        out = self.decode(z)
        return out, mu, log_var


networks = {
    "SimpleVAE": SimpleVAE,
    "SimpleConditionalVAE": SimpleConditionalVAE,
    "FullyConnectedVAE": FullyConnectedVAE,
}

# # model.py ends here
