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
    Training process

    :param device: torch.device
        The device to train network on
    :param n_epochs: int
        Number of epochs during training
    :param data_path: str
        The path to the data
    :param train_test_split: float
        The proportion of the training set
    :param net: str
        The network to train, 'cnn' for CNN, 'rnn' for RNN, 'both' for both
    :param resume_training: bool
        True, resume training from previously saved checkpoint, False otherwise.

    :return: None
    """

    train_on(net, device)

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

    # For training statistics
    total = 0

    # Training loop
    for epoch in range(n_epochs):
        # print(torch.cuda.memory_summary(device=device, abbreviated=False))

        # training loss tracker
        running_loss = []

        # progress bar
        # p_bar = tqdm(total=len(train_data), bar_format='{l_bar}{bar:20}{r_bar}',
        #              unit=' batches', ncols=200, mininterval=0.02, colour='#00ff00')

        for i, (images, labels_d, anchors, label) in enumerate(train_data):
            # unpacking
            images, labels_d, anchors, label = images.to(device), labels_d.to(device), anchors.to(device), label.to(
                device)

            # initialize gradients
            optimizer.zero_grad()

            # forward pass
            if net == 'cnn':
                output_d, _ = model(images, False, anchors)

                # compute the loss
                loss = criterion(torch.transpose(output_d, 1, 3), labels_d)
            else:
                _, output_f = model(images, False, anchors)
                # compute the loss
                loss = criterion(output_f, torch.zeros((5, 1, 2, 2), dtype=torch.float32).to(device))

            # backward propagation
            loss.backward(retain_graph=True)

            # DONE: ADD CLIP GRADIENT FOR STABLE NETWORK
            # DONE: from torch.nn.utils import clip_grad_norm_
            # DONE: clip_grad_norm_(model.parameters(), 1)
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

            if net == "cnn":
                optimizer.step()

            # compute statistics
            total += labels_d.size(0)

            # accumulate loss values
            running_loss.append(loss.item())

            # update the progress bar
            # p_bar.set_postfix(
            #     epoch=f"{epoch}/{n_epochs}, train loss= {round(sum(running_loss) / len(running_loss), 2)}",
            #     refresh=True)

            # update the progress bar
            # p_bar.update()
            print(f'Epoch: {epoch}/{n_epochs}, iteration: {i}/{len(train_data)}, loss: {loss.item()}')
        # close progress bar
        # p_bar.close()

        # saving model/models
        save_checkpoints(model, epoch)


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
                images, labels_d, anchors, label = images.to(device), labels_d.to(device), anchors.to(device), label.to(
                    device)

                # For training statistics
                cnn_running_loss = []
                rnn_running_loss = []

                # initialize gradients
                optimizer.zero_grad()

                output_d, output_f = model(images, False, anchors)

                # compute the cnn loss
                cnn_loss = criterion(torch.transpose(output_d, 1, 3), labels_d)

                label = sample_label(label, device)

                # compute the rnn loss
                rnn_loss = criterion(output_f, label)

                # backward propagations
                cnn_loss.backward(retain_graph=True)
                rnn_loss.backward(retain_graph=True)

                # TODO: Something Else


                # accumulate loss values
                cnn_running_loss.append(cnn_loss.item())
                rnn_running_loss.append(rnn_loss.item())

                # update the progress bar
                p_bar.set_postfix(
                    epoch=f"{epoch}/{n_epochs}, "
                          f"cnn_loss: {round(sum(cnn_running_loss) / len(cnn_running_loss), 4)}, "
                          f"rnn_loss: {round(sum(rnn_running_loss) / len(rnn_running_loss), 4)}",
                    refresh=True)

                # update the progress bar
                p_bar.update()

                # print(f'Epoch: {epoch}/{n_epochs}, iteration: {i}/{len(train_data)}, '
                #       f'cnn_loss: {round(sum(cnn_running_loss) / len(cnn_running_loss), 4)}, '
                #       f' rnn_loss: {round(sum(rnn_running_loss) / len(rnn_running_loss), 4)}')

        if epoch % 10 == 0:
            # saving model
            save_checkpoints(model, epoch)
