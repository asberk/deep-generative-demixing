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
        self, in_features, hidden_features, latent_features, device=None
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

        self.in_features = in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc21 = nn.Linear(hidden_features, latent_features)
        self.fc22 = nn.Linear(hidden_features, latent_features)
        self.fc3 = nn.Linear(latent_features, hidden_features)
        self.fc4 = nn.Linear(hidden_features, in_features)

    def encode(self, input):
        h1 = torch.relu(self.fc1(input))
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
        h3 = torch.relu(self.fc3(z))
        out = torch.sigmoid(self.fc4(h3))
        return out

    def forward(self, input):
        mu, log_var = self.encode(input.view(-1, self.in_features))
        z = self.reparametrize(mu, log_var)
        out = self.decode(z)
        return out, mu, log_var


# # model.py ends here
