from torch.utils.data import Dataset
from typing import Tuple
from torch.utils.data import Subset
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

    train_set = Subset(dataset, range(train_length))
    test_set = Subset(dataset, range(train_length, train_length + test_length))

    return train_set, test_set


if __name__ == "__main__":
    # dataset
    dataset_npz = NPZLoader('./Data')

    train, test = get_sets(dataset_npz, 0.2)
