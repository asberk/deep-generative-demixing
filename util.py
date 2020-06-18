import re
from datetime import datetime
from collections import defaultdict
import pandas as pd


class Logger:
    def __init__(self):
        self.log = defaultdict(list)

    def __call__(self, key, value):
        self.log[key].append(value)

    def __repr__(self):
        keys = ", ".join(f"{key}" for key in self.log.keys())
        return f"Logger({keys})"

    def __getitem__(self, idx):
        return self.log[idx]

    def to_df(self):
        return pd.DataFrame(self.log)

    def save(self, filepath, **kwargs):
        """
        save(self, filepath, **kwargs)

        Parameters
        ----------
        filepath: string
        kwargs: dict
            keyword arguments to pd.DataFrame.to_csv
        """
        self.to_df().to_csv(filepath, **kwargs)
        print(f"Saved log to\n  {filepath}")

    def clear(self):
        self.log = defaultdict(list)


def get_tstamp():
    tstamp = (
        datetime.now()
        .isoformat()
        .replace("-", "")
        .replace("T", "-")
        .replace(":", "")
        .replace(".", "-")
    )
    return tstamp


def save_args(args, fpath=None):
    import csv

    if fpath is None:
        fpath = f"./log/args_{get_tstamp()}.csv"

    argdict = args.__dict__
    col_names = list(argdict.keys())
    try:
        with open(fpath, "w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=col_names)
            writer.writeheader()
            writer.writerow(argdict)
    except IOError as ioe:
        print(f"IOError: {ioe}")
