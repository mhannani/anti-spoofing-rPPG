import torch.nn as nn
import torch.optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.NPZ_loader import NPZLoader
from models.Cnn_Rnn import CnnRnn
from utils.split import get_sets
from utils.save import save_checkpoints
from utils.sample import sample_label
from utils.show import train_on
from utils.create import create_network

torch.autograd.set_detect_anomaly(True)


def train(device: torch.device, n_epochs: int = 1000, data_path: str = './Data',
          train_test_split: float = 0.06, net: str = 'cnn', resume_training: bool = True):
    """
    Not implemented.
    """
    raise NotImplementedError


def train_all(device: torch.device, n_epochs: int = 1000, data_path: str = './Data',
              train_test_split: float = 0.8, resume_training: bool = True):
    """
    Train the CNN_RNN network.
    """

    # Get data
    dataset = NPZLoader(data_path)

    # split dataset
    train_set, _ = get_sets(dataset, train_test_split)

    # Get train dataloader
    train_data = DataLoader(train_set, batch_size=5)

    model = create_network(resume_training, device)

    # loss function
    criterion = nn.MSELoss()

    # define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, betas=(0.9, 0.999), eps=1e-08)

    # Training loop
    for epoch in range(n_epochs):
        # progress bar
        p_bar = tqdm(total=len(train_data), bar_format='{l_bar}{bar:20}{r_bar}',
                     unit=' batches', ncols=200, mininterval=0.02, colour='#00ff00')

        for i, (images, labels_d, anchors, label) in enumerate(train_data):
            # Send variables to device
            images = images.to(device)
            labels_d = labels_d.to(device)
            anchors = anchors.to(device)
            label = label.to(device)

            # initialize gradients
            optimizer.zero_grad()
            # print('image shape: ', images.shape)
            output_d, output_f = model(images, False, anchors)

            # compute the cnn loss
            # print('label_d: ', labels_d.shape)
            cnn_loss = criterion(torch.transpose(output_d, 1, 3), labels_d)

            label = sample_label(label, device)

            # compute the rnn loss
            rnn_loss = criterion(output_f, label)

            # backward propagations
            cnn_loss.backward(retain_graph=True)
            rnn_loss.backward(retain_graph=True)

            optimizer.step()

            # accumulate loss values
            cnn_running_loss = cnn_loss.item()
            rnn_running_loss = rnn_loss.item()

            # update the progress bar
            p_bar.set_postfix(
                epoch=f"{epoch}/{n_epochs}, "
                      f"cnn_loss: {round(cnn_running_loss / 5, 4)}, "
                      f"rnn_loss: {round(rnn_running_loss / 5, 4)}",
                refresh=True)

            # update the progress bar
            p_bar.update()

        if epoch % 3 == 0:
            # saving model
            save_checkpoints(model, epoch)
