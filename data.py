import numpy as np
import torch
from torch.utils.data import Subset, Dataset, DataLoader
from torchvision import datasets, transforms

DATA_DIR = "./data/"


class MNISTSubset(datasets.mnist.MNIST):
    def __init__(
        self,
        root,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
        classes=None,
    ):
        super().__init__(root, train, transform, target_transform, download)

        if classes is not None:
            if not isinstance(classes, (tuple, list)):
                classes = [classes]
            assert all(
                isinstance(x, np.int) for x in classes
            ), f"Expected int or iterable of ints but got {classes}"

            indices = [i for i, n in enumerate(self.targets) if n in classes]
            self.targets = self.targets[indices]
            self.data = self.data[indices]
            self._reset_classes(classes)

    def _reset_classes(self, classes):
        self.classes = [
            entry for entry in self.classes if int(entry[0]) in classes
        ]
        class_2_idx = {
            key: value
            for key, value in self.class_to_idx.items()
            if key in self.classes
        }
        self.class_2_idx = {key: i for i, key in enumerate(class_2_idx.keys())}


def load_mnist_datasets(data_dir=None, transform=None, classes=None):
    if data_dir is None:
        data_dir = DATA_DIR
    if transform is None:
        transform = transforms.ToTensor()

    dset_train = MNISTSubset(
        data_dir,
        train=True,
        transform=transform,
        download=True,
        classes=classes,
    )
    dset_test = MNISTSubset(
        data_dir,
        train=False,
        transform=transform,
        download=True,
        classes=classes,
    )
    return dset_train, dset_test


def _size_helper(subset_size, parent_size, name, default_size=0.2):
    if subset_size is None:
        subset_size = default_size
    assert subset_size > 0, f"Expected {name} to be positive"
    if isinstance(subset_size, np.float):
        subset_size = int(parent_size * subset_size)
    else:
        assert isinstance(
            subset_size, np.int
        ), f"Expected {name} to be int or float but got {subset_size}"
    assert (
        subset_size < parent_size
    ), f"Expected {name} < {parent_size} but found {subset_size}"
    return subset_size


def get_partitioned_datasets(
    dset_dev,
    dset_holdout=None,
    phases=None,
    train_eval_size=None,
    val_size=None,
):
    if phases is None:
        phases = ["train", "train_eval", "val", "test"]

    ret = {}
    if "test" in phases:
        assert (
            dset_holdout is not None
        ), f"Expected dset_holdout if 'test' in `phases`."
        ret["test"] = dset_holdout

    dev_size = len(dset_dev)

    if "val" in phases:
        val_size = _size_helper(val_size, dev_size, "val_size", 0.2)
        val_indices = np.random.choice(dev_size, val_size, replace=False)
        train_indices = np.setdiff1d(range(dev_size), val_indices)
        train_set = Subset(dset_dev, train_indices)
        val_set = Subset(dset_dev, val_indices)
        ret["train"] = train_set
        ret["val"] = val_set

    else:
        train_indices = range(dev_size)
        train_set = dset_dev
        ret["train"] = train_set

    if "train_eval" in phases:
        train_size = len(train_set)
        train_eval_size = _size_helper(
            train_eval_size, train_size, "train_eval_size", 0.5
        )
        train_eval_indices = np.random.choice(
            train_indices, size=train_eval_size, replace=False
        )
        train_eval_set = Subset(dset_dev, train_eval_indices)
        ret["train_eval"] = train_eval_set

    for key in ret.keys():
        if isinstance(ret[key], Subset):
            setattr(ret[key], "classes", ret[key].dataset.classes)
            setattr(ret[key], "class_2_idx", ret[key].dataset.class_2_idx)
            setattr(ret[key], "data", ret[key].dataset.data)
            setattr(ret[key], "targets", ret[key].dataset.targets)

    return ret


def _parse_batch_size_shuffle(batch_size, shuffle):
    if batch_size is None:
        batch_size = {}
    elif isinstance(batch_size, np.int):
        batch_size = {"train": batch_size}
    assert isinstance(
        batch_size, dict
    ), f"Expected dict for batch_size but found {batch_size}."
    batch_size.setdefault("train", 128)
    for phase in ["train_eval", "val", "test"]:
        batch_size.setdefault(phase, 128)
    if shuffle is None:
        shuffle = {}
    elif isinstance(shuffle, bool):
        shuffle = {"train": shuffle}
    assert isinstance(
        shuffle, dict
    ), f"Expected dict for shuffle but found {shuffle}."
    shuffle.setdefault("train", True)
    for phase in ["train_eval", "val", "test"]:
        shuffle.setdefault(phase, False)
    return batch_size, shuffle


def get_dataloaders(datasets, batch_size=None, shuffle=None):
    batch_size, shuffle = _parse_batch_size_shuffle(batch_size, shuffle)
    assert isinstance(
        datasets, dict
    ), f"Expected datasets to be a dict but found {type(datasets)}"

    loaders = {
        phase: DataLoader(
            dataset, batch_size=batch_size[phase], shuffle=shuffle[phase]
        )
        for phase, dataset in datasets.items()
    }
    return loaders


def basic_1_8_setup(ravel=True, batch_size=128):
    """
    basic_1_8_setup()

    Datasets and dataloaders with the digits 1 and 8 only.

    Returns
    -------
    datasets : dict
        keys: ["train", "train_eval", "val", "test"]
        proportions: [ 80% dev, 50% train, 20% dev, 100% holdout ]
    dataloaders : dict
        batch_size: {"train" : 16, "train_eval": 128, "val": 128, "test": 128}
        shuffle: {"train" : True, "train_eval": False, "val": False, "test": False}
    """
    if ravel:
        on_load_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))]
        )
    else:
        on_load_transform = transforms.ToTensor()
    dset_dev, dset_ho = load_mnist_datasets(
        transform=on_load_transform, classes=[1, 8]
    )
    datasets = get_partitioned_datasets(dset_dev, dset_ho)
    dataloaders = get_dataloaders(datasets, batch_size=batch_size)
    img_shape = datasets["train"][0][0].size()
    classes = datasets["train"].targets.unique()
    return dataloaders, img_shape, classes


load_data_fns = {"basic_1_8_setup": basic_1_8_setup}
