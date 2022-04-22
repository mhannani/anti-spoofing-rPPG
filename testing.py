import torch.nn as nn
import torch
from torch.utils.data import DataLoader

from models.Cnn_Rnn import CnnRnn
from utils.NPZ_loader import NPZLoader
from utils.split import get_sets
from utils.read import get_device, read_config_yaml


def train_RNN(net, optimizer, trainloader, criterion, n_epoch=10):
    total = 0
    for epoch in range(n_epoch):
        # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # Donnees pre-crees:
            images, labels_d, anchors, label = data
            # training step
            optimizer.zero_grad()
            _, outputs_F = net(images, False, anchors)

            loss = criterion(outputs_F, outputs_F)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=10)
            # optimizer.step()

            # compute statistics
            total += labels_d.size(0)
            running_loss += loss.item()
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / total))
        print('Epoch finished')
    print('Finished Training')


if __name__ == "__main__":
    # Get data
    dataset = NPZLoader("./Data")

    # split dataset
    train_set, _ = get_sets(dataset, 0.8)

    # Get train dataloader
    train_data = DataLoader(train_set, batch_size=5)

    cfg = read_config_yaml('config.yaml')
    device = get_device(cfg)
    model = CnnRnn(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, betas=(0.9, 0.999), eps=1e-08)

    train_RNN(model, optimizer, train_data, criterion)
