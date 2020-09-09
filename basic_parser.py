"""
basic_parser

Used for gd_numerics_two_network.py

Author: Aaron Berk <aberk@math.ubc.ca>
Copyright © 2020, Aaron Berk, all rights reserved.
Created:  9 September 2020
"""
from argparse import ArgumentParser

parser = ArgumentParser(
    description="Train two VAEs as generators, each on a separate MNIST digit."
)
parser.add_argument(
    "--digit1", type=int, default=1, help="first digit class (default: 1)"
)
parser.add_argument(
    "--digit2", type=int, default=8, help="second digit class (default: 8)"
)
parser.add_argument(
    "--train-batch-size",
    type=int,
    default=32,
    help="batch size for train data default=32",
)
parser.add_argument(
    "--val-batch-size",
    type=int,
    default=128,
    help="batch size for val data default=128",
)
parser.add_argument(
    "--network",
    type=str,
    default="SimpleVAE",
    help="network type. Options include SimpleVAE, FullyConnectedVAE, SimpleConditionalVAE", "CNN_VAE",
)
parser.add_argument(
    "--network-hidden-features", type=int, default=None, help="",
)
parser.add_argument(
    "--network-latent-features", type=int, default=None, help="",
)
parser.add_argument(
    "--network-device", type=str, default=None, help="",
)
parser.add_argument(
    "--network-dropout-probability",
    type=float,
    default=None,
    help="",
)
parser.add_argument(
    "--criterion", type=str, default="default", help="default='default'",
)
parser.add_argument(
    "--criterion-lamda", type=float, default=1.0, help="default=1.0",
)
parser.add_argument(
    "--optimizer", type=str, default="SGD", help="default='SGD'",
)
parser.add_argument(
    "--optimizer-lr", type=float, default=1e-5, help="default=None",
)
parser.add_argument(
    "--optimizer-momentum", type=float, default=0.9, help="default=0.9",
)
parser.add_argument(
    "--optimizer-weight-decay", type=float, default=1e-3, help="default=1e-3",
)
parser.add_argument(
    "--epochs", type=float, default=200, help="default=200",
)
parser.add_argument(
    "--auto-lr", type=bool, default=True, help="default=True",
)
parser.add_argument(
    "--train", type=bool, default=True, help="default=True",
)

# # basic_parser.py ends here
