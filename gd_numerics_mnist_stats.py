"""
GD_NUMERICS_MNIST_STATS

Interpreting parameter settings and model set-up for the MNIST-MNIST experiment

Author: Aaron Berk <aberk@math.ubc.ca>
Copyright © 2020, Aaron Berk, all rights reserved.
Created: 23 September 2020
"""
import os
from glob import glob
import shutil
import pandas as pd
from util import search_for_tstamp_directory, pretty_print_args_csv
from viz import plot_val_log_from_tstamp

tstamps = ["20200914-105758-237836", "20200914-105759-069972"]
latex_dir = os.path.join(
    os.path.expanduser("~"),
    "Dropbox/school/phd/research/notes/generative-demixing/fig",
)


def copy_over_eval_imgs(tstamps, latex_dir):
    src_dirs = [search_for_tstamp_directory(tstamp) for tstamp in tstamps]
    common_path = os.path.commonpath(src_dirs)
    parent_tstamp = os.path.split(common_path)[-1]

    for tstamp, src_dir in zip(tstamps, src_dirs):
        eval_img_fpaths = glob(os.path.join(src_dir, "eval_img", "img_*.jpg"))
        final_epoch = max(
            [
                int(os.path.split(eval_img_fpath)[-1].split("_")[1])
                for eval_img_fpath in eval_img_fpaths
            ]
        )
        eval_img_fpaths = glob(
            os.path.join(src_dir, "eval_img", f"img_{final_epoch}_*.jpg")
        )
        dest_dir = os.path.join(latex_dir, tstamp)

        for eval_img_fpath in eval_img_fpaths:
            eval_img_fname = os.path.split(eval_img_fpath)[-1]
            src_fpath = os.path.join(src_dir, "eval_img", eval_img_fname)
            dest_fpath = os.path.join(dest_dir, eval_img_fname)
            shutil.copyfile(src_fpath, dest_fpath)


def write_pretty_args(tstamp):
    directory = search_for_tstamp_directory(tstamp)
    fname = "args_pretty.txt"
    with open(os.path.join(directory, fname), "w") as fp:
        fp.write(f"{tstamp}\n")
        pretty_print_args_csv(tstamp, file=fp)


def copy_over_files(tstamps, latex_dir):
    src_dirs = [search_for_tstamp_directory(tstamp) for tstamp in tstamps]
    common_path = os.path.commonpath(src_dirs)
    parent_tstamp = os.path.split(common_path)[-1]

    print(f"Warning: not using {parent_tstamp} at dest.")
    dest_dirs = [os.path.join(latex_dir, tstamp) for tstamp in tstamps]

    fnames = ["val_log_plot.pdf", "args_pretty.txt"]
    for src_dir, dest_dir in zip(src_dirs, dest_dirs):
        for fname in fnames:
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            src_fpath = os.path.join(src_dir, fname)
            dest_fpath = os.path.join(dest_dir, fname)
            if not os.path.exists(dest_fpath):
                shutil.copyfile(src_fpath, dest_fpath)


for tstamp in tstamps:
    print(f"\n{tstamp}")

    plot_val_log_from_tstamp(tstamp)
    write_pretty_args(tstamp)
    print("done")

copy_over_files(tstamps, latex_dir)
copy_over_eval_imgs(tstamps, latex_dir)

# # gd_numerics_mnist_stats.py ends here
