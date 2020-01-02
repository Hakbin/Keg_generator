import os

import numpy as np
import pandas as pd
import torch
import torchvision.datasets as torch_datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler, TensorDataset

ROOT_PATH = '../data'


def _normalize(arr):
    """
    Normalize a numpy array into zero-mean and unit-variance.
    """
    avg = arr.mean(axis=0)
    std = arr.std(axis=0)
    arr = arr-avg
    arr[:, std != 0] /= std[std != 0]
    return arr


def _split2(nd, seed=2019):
    """
    Split data into the 7:1 ratios.
    """
    shuffled_index = np.arange(nd)
    np.random.seed(seed)
    np.random.shuffle(shuffled_index)
    size = int(nd * 7 / 8)
    index1 = shuffled_index[:size]
    index2 = shuffled_index[size:]
    return index1, index2


def _split3(nd, seed=2019):
    """
    Split data into the 7:1:2 ratios.
    """
    shuffled_index = np.arange(nd)
    np.random.seed(seed)
    np.random.shuffle(shuffled_index)
    index1 = shuffled_index[:int(nd * 0.7)]
    index2 = shuffled_index[int(nd * 0.7):int(nd * 0.8)]
    index3 = shuffled_index[int(nd * 0.8):]
    return index1, index2, index3


def _get_samplers(num_data, num_valid_data, seed):
    """
    Return a pair of samplers for the image datasets.
    """
    indices = np.arange(num_data)
    np.random.seed(seed)
    np.random.shuffle(indices)

    train_sampler = SubsetRandomSampler(indices[:-num_valid_data])
    valid_sampler = SubsetRandomSampler(indices[-num_valid_data:])
    return train_sampler, valid_sampler


def _to_image_loaders(trn_data, val_data, test_data, batch_size):
    """
    Convert an image dataset into loaders.
    """
    samplers = _get_samplers(len(trn_data), 5000, seed=2019)
    trn_l = DataLoader(trn_data, batch_size, sampler=samplers[0])
    val_l = DataLoader(val_data, batch_size, sampler=samplers[1])
    test_l = DataLoader(test_data, batch_size)
    return trn_l, val_l, test_l


class MNIST:
    """
    Class for the MNIST dataset.
    """
    nx = 1024
    ny = 10
    nc = 1
    size = 1, 32, 32

    @staticmethod
    def to_loaders(batch_size):
        """
        Convert the dataset into data loaders.
        """
        path = f'{ROOT_PATH}/mnist'
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        trn_data = torch_datasets.MNIST(
            path, train=True, transform=transform, download=True)
        test_data = torch_datasets.MNIST(
            path, train=False, transform=transform)
        return _to_image_loaders(trn_data, trn_data, test_data, batch_size)


def to_dataset(dataset):
    """
    Return a dataset class given its name.
    """
    if dataset == 'mnist':
        return MNIST()