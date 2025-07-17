"""
FICDataset Class adjusted from the fashion_captioning repo
https://github.com/xuewyang/Fashion_Captioning/blob/da6ab40fe09a9b1331afbe3055b7d8137f415e17/dataset.py#L9
"""

import h5py
import json
import os
import torch
from torch.utils.data import Dataset, Subset
from datasets import load_from_disk, Dataset
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()
fic_dataset = os.getenv("FACAD")
hm_dataset = os.getenv("HM_SPLIT_ARROW")


class FICDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, split, data_folder=fic_dataset, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'train', 'val', or 'test'
        :param transform: image transform pipeline
        """
        self.data_split = split.upper()
        assert self.data_split in {"TRAIN", "VAL", "TEST"}
        print(f"Loading {self.data_split.lower()} split for FACAD")

        self.h = h5py.File(os.path.join(data_folder, self.data_split + "_IMAGES" + ".hdf5"), "r")
        self.imgs = self.h["images"]

        with open(os.path.join(data_folder, self.data_split + "_CAPTIONS_RAW" + ".json"), "r") as j:
            self.captions = json.load(j)

        with open(os.path.join(data_folder, self.data_split + "_ATTRS" + ".json"), "r") as j:
            self.attrs = json.load(j)

        with open(os.path.join(data_folder, self.data_split + "_CATES_FLAT" + ".json"), "r") as j:
            self.cates = json.load(j)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform
        self.size = len(self.captions)

    def __getitem__(self, i):
        """
        :returns: Properties of ith item as Tuple (image, caption, caption length)
        :rtype: Tuple (Tensor, Tensor, Tensor)
        """
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        # img = torch.FloatTensor(self.imgs[i / 255.])
        img = torch.FloatTensor(self.imgs[i])
        if self.transform is not None:
            img = self.transform(img)

        caption = self.captions[i]
        # cate = self.cates[i]
        # attr = torch.LongTensor(self.attrs[i])

        return {"image": img, "text": caption, "id": i}

    def __getitems__(self, indices):
        return [self.__getitem__(i) for i in indices]

    def plot_image(self, i, filename=None):
        """
        :param filename: Default None, if value given the image will be saved
        """
        img = self.imgs[i].transpose((1, 2, 0))
        p_img = Image.fromarray(img)
        plt.imshow(p_img)
        if filename:
            plt.savefig(filename)

    def __len__(self):
        return self.size


class HMDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, split, data_folder=hm_dataset, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'train', 'val', or 'test'
        :param transform: image transform pipeline
        """
        # if split then check
        assert split in {"train", "val", "test"}
        self.data_split = split.replace("val", "validation") if "val" in split else split

        print(f"Loading {self.data_split} split for HM")
        self.dataset = load_from_disk(os.path.join(data_folder, self.data_split))

        # with open(os.path.join(data_folder, self.split + '_categories' + '.json'), 'r') as j:
        #    self.cates = json.load(j)

        # TODO
        # with open(os.path.join(data_folder, self.split + '_attrs' + '.json'), 'r') as j:
        #    self.attrs = json.load(j)

        self.size = len(self.dataset)
        self.transform = transform

    def __getitem__(self, i):
        """
        :returns: dict of single item with "image" and "text" keys for the respective values
        :rtype: img - PIL.Image.Image, caption - str
        """
        id = self.dataset[i]["filename"]
        img = self.dataset[i]["image"]
        caption = self.dataset[i]["text"]
        # cate = self.cates[i]
        # attr = torch.LongTensor(self.attrs[i])

        return {"image": img, "text": caption, "id": id}

    def plot_item(self, i, filename=None):
        """
        Plots item and text description.
        :param filename: Default None, if value given the image will be saved
        """
        example = self.dataset[i]
        image = example["image"]
        width, height = image.size
        print(example["text"])
        display(image.resize((int(0.3 * width), int(0.3 * height))))  # type: ignore
        if filename:
            plt.savefig(filename)

    def __len__(self):
        return self.size


def get_subset(dataset, num: int):
    """
    Returns a subset of the dataset with the first `num` items.
    :param dataset: The dataset to subset.
    :param num: The number of items to include in the subset.
    :return: A Subset of the dataset containing the first `num` items.
    """
    return Subset(dataset, range(num))


def get_datasets(dataset: str, test: bool = False, num: int = 100, debug: bool = False):
    """
    Returns the appropriate dataset based on the specified parameters.
    :param dataset: The name of the dataset to load, either "hm" or "fic".
    :param test: If True, returns the test split of the dataset; otherwise, returns the train and validation splits.
    :param num: The number of items to include in the subset if debug is True.
    :param debug: If True, returns a subset of the dataset with the first `num` items.
    :return: The requested dataset or its subsets.
    """
    assert dataset in ["hm", "fic"]
    if dataset == "hm":
        if test:
            hm_test = HMDataset("test")
            if debug:
                return get_subset(hm_test, num)
            return hm_test
        else:
            hm_train = HMDataset("train")
            hm_val = HMDataset("val")
            if debug:
                return get_subset(hm_train, num), get_subset(hm_val, num)  # EXPERIMENT, REMOVE
            return hm_train, hm_val
    elif dataset == "fic":
        if test:
            fic_test = FICDataset("test")
            if debug:
                return get_subset(fic_test, num)
            return fic_test
        else:
            fic_train = FICDataset("train")
            fic_val = FICDataset("val")
            if debug:
                return get_subset(fic_train, num), get_subset(fic_val, num)
            return fic_train, fic_val
