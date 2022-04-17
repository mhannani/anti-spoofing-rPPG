from torch.utils.data import DataLoader
from utils.NPZ_loader import NPZLoader


def train(n_epochs: int, data_path: str):
    """
    Training process.
    :param n_epochs: int
        Number of epochs during training
    :param data_path: str
        The path to the data
    :return: None
    """

    # Get data
    data = NPZLoader(data_path)

    # Get dataloader
    train_data = DataLoader(data, batch_size=5)

    # Training loop
    for epoch in range(n_epochs):
        for i, data in enumerate(train_data):
            pass

    raise NotImplemented('Not yet finished...')
