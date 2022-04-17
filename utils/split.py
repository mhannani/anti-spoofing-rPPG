from torch.utils.data import Dataset
from typing import Tuple
from torch.utils.data import random_split
from utils.NPZ_loader import NPZLoader


def get_sets(dataset: NPZLoader, train_test_split: float) -> Tuple[Dataset, Dataset]:
    """
    Splits and returns the dataset

    :param dataset: Dataset
        The dataset object from torch.utils.data.Dataset
    :param train_test_split: float
        The proportion of the training
    return: Tuple
        The two sets: (train_set, test_set)
    """

    # compute the length of sets
    train_length = int(train_test_split * dataset.__len__())
    test_length = int(dataset.__len__() - train_length)

    # split the dataset
    train_set, test_set = random_split(dataset, [train_length, test_length])

    return train_set, test_set
