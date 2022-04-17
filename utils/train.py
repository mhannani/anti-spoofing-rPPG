import torch.nn as nn
import torch.optim

from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.NPZ_loader import NPZLoader
from models.Cnn_Rnn import CnnRnn


def train(n_epochs: int = 10, data_path: str = './Data', net: str = 'cnn'):
    """
    Training process

    :param n_epochs: int
        Number of epochs during training
    :param data_path: str
        The path to the data
    :param net: str
        The network to train, 'cnn' for CNN, 'rnn' for RNN, 'both' for both
    :return: None
    """

    if net not in ['cnn', 'rnn', 'both']:
        raise NotImplemented('Net value is not implemented')

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
        # training loss tracker
        running_loss = []

        # progress bar
        p_bar = tqdm(total=len(train_data), bar_format='{l_bar}{bar:10}{r_bar}',
                     unit=' batches', ncols=200, mininterval=0.05, colour='#00ff00')

        for i, data in enumerate(train_data):
            # unpacking
            images, labels_d, anchors, _ = data

            # initialize gradients
            optimizer.zero_grad()

            # forward pass
            output_d, _ = model(images, False, anchors)

            # compute the loss
            loss = criterion(torch.transpose(output_d, 1, 3), labels_d)

            # backward propagation
            loss.backward()

            # TODO: ADD CLIP GRADIENT FOR STABLE NETWORK
            # TODO: from torch.nn.utils import clip_grad_norm_
            # TODO: clip_grad_norm_(model.parameters(), 1)

            # compute statistics
            total += labels_d.size(0)

            # accumulate loss values
            running_loss.append(loss.item())

            # update the progress bar
            p_bar.set_postfix(
                epoch=f" {epoch}/{n_epochs}, train loss= {round(sum(running_loss) / len(running_loss), 2)}",
                refresh=True)

            # update the progress bar
            p_bar.update()

        # close progress bar
        p_bar.close()

