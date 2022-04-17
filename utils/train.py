import torch.optim
from torch.utils.data import DataLoader
from utils.NPZ_loader import NPZLoader
import torch.nn as nn

# to be modified
from models.Cnn_Rnn import CnnRnn


def train(n_epochs: int = 10, data_path: str = './Data', net: str = 'cnn'):
    """
    Training process

    :param n_epochs: int
        Number of epochs during training
    :param data_path: str
        The path to the data
    :param net: str
        The network to train, 'cnn' for CNN, 'rnn' for RNN
    :return: None
    """

    # Get data
    dataset = NPZLoader(data_path)

    # Get dataloader
    train_data = DataLoader(dataset, batch_size=5)

    # model
    model = CnnRnn()

    # loss function
    criterion = nn.MSELoss()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, betas=(0.9, 0.999), eps=1e-08)

    # for training statistics
    total = 0

    # Training loop
    for epoch in range(n_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_data):
            # unpacking
            images, labels_d, anchors, _ = data

            # initialize gradients
            optimizer.zero_grad()

            # forward pass
            output_d, _ = model(images, False, anchors)

            # compute the loss
            loss = criterion(output_d, labels_d)

            # backward propagation
            loss.backward(retain_graph=True)

            # compute statistics
            total += labels_d.size(0)

            # accumulate loss values
            running_loss += loss.item()

    raise NotImplemented('Not yet finished...')
