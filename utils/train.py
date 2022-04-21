import torch.nn as nn
import torch.optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.NPZ_loader import NPZLoader
from models.Cnn_Rnn import CnnRnn
from utils.split import get_sets
from utils.save import save_checkpoints
from utils.load import check_saved_checkpoints, load_last_checkpoints


def train(device: torch.device, n_epochs: int = 10, data_path: str = './Data', train_test_split: float = 0.8, net: str='cnn', resume_training: bool = True):
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

    if device == 'cpu':
        print(f'Training {net} on cpu...')
    else:
        torch.cuda.empty_cache()
        print(f'Training {net} on gpu...')

    if net not in ['cnn', 'rnn']:
        raise NotImplemented('Net value is not implemented')

    # if net == 'cnn':
    #     pass
    #     print('Training cnn...')
    # else:
    #     pass
    #     print('Training rnn...')

    # Get data
    dataset = NPZLoader(data_path)

    # split dataset
    train_set, _ = get_sets(dataset, train_test_split)

    # Get train dataloader
    train_data = DataLoader(train_set, batch_size=5)

    if resume_training and check_saved_checkpoints('./pretrained'):
        print('Loading saved model and resume training...')
        model = load_last_checkpoints('./pretrained/').to(device)
    else:
        print('No checkpoint to resume training from... Training from scratch.')
        model = CnnRnn()
        model = model.to(device)

    # loss function
    criterion = nn.MSELoss()

    # define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, betas=(0.9, 0.999), eps=1e-08)

    # for training statistics
    total = 0

    # Training loop
    for epoch in range(n_epochs):
        # training loss tracker
        running_loss = []

        # progress bar
        p_bar = tqdm(total=len(train_data), bar_format='{l_bar}{bar:20}{r_bar}',
                     unit=' batches', ncols=200, mininterval=0.02, colour='#00ff00')
        for i, (images, labels_d, anchors, label) in enumerate(train_data):
            # unpacking
            images, labels_d, anchors, label = images.to(device), labels_d.to(device), anchors.to(device), label.to(device)

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
                loss = criterion(output_f, torch.zeros((5, 1, 2), dtype=torch.float32))

            # backward propagation
            loss.backward()

            # Performs single optimization step
            optimizer.step()

            # TODO: ADD CLIP GRADIENT FOR STABLE NETWORK
            # TODO: from torch.nn.utils import clip_grad_norm_
            # TODO: clip_grad_norm_(model.parameters(), 1)

            # compute statistics
            total += labels_d.size(0)

            # accumulate loss values
            running_loss.append(loss.item())

            # update the progress bar
            p_bar.set_postfix(
                epoch=f"{epoch}/{n_epochs}, train loss= {round(sum(running_loss) / len(running_loss), 2)}",
                refresh=True)

            # update the progress bar
            p_bar.update()

        # close progress bar
        p_bar.close()

        # saving model/models
        save_checkpoints(model, epoch)
