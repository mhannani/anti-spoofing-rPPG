import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split


class NPZLoader(Dataset):
    """
    Dataset from the NPZ files.

    Prevent loading all the npz files into memory for efficient
    training process.
    """

    def __init__(self, path):
        """
        The class init function

        :param path: str
            Absolute path to the data folder.
        :return: None
        """

        # Get the path/folder path
        self.path = path

        # Get all files with nzp extension
        self.files = list(sorted(Path(path).glob('**/*.npz')))

        # Get anchors from npz fil
        self.anchors_npz = np.load(str(self.files[0]))

        # Get images from npz file
        self.images_npz = np.load(str(self.files[1]))

        # Get label from npz file
        self.label_npz = np.load(str(self.files[2]))

        # Get label_d from npz file
        self.label_d_npz = np.load(str(self.files[3]))

    def __len__(self):
        """
        Returns the length of the dataset.

        :return: int
            The length of the dataset.
        """

        return len(self.images_npz)

    def __getitem__(self, item):
        """
        Get the item nth in the Dataset.

        :param item: int
            The index of the element.
        :return: None
        """

        return np.transpose(self.images_npz['{}'.format(item)].astype(np.float32), (2, 0, 1)), \
            self.label_d_npz['{}'.format(item)].astype(np.float32), \
            self.anchors_npz['{}'.format(item)].astype(np.float32), \
            self.label_npz['{}'.format(item)].astype(np.float32)

    def __call__(self):
        """
        Call the class a function.
        :return: None
        """

        print(self.__getitem__(0)[0].shape)


if __name__ == "__main__":
    # Just for testing purposes
    npz_loader = NPZLoader('../Data/')
    npz_loader()

    # split the dataset
    train_set, test_set = random_split(npz_loader, [100, 1900])

    # testing with dataloader
    data_loader = DataLoader(train_set, batch_size=5)

    # check the returned structure
    for i, data in enumerate(data_loader):
        print(i)
        a, b, c, d = data
        print(a.shape, b.shape, c.shape, d.shape)
        break
