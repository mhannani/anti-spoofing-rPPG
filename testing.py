import numpy as np
import torch
from models.Anti_Spoof_net import Anti_spoof_net


def train_CNN(net, optimizer, trainloader, data_anchors, criterion, n_epoch = 10):
    total = 0
    for epoch in range(n_epoch):
        # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            #Pre-created Data:
            images, labels_D = data
            # training step
            optimizer.zero_grad()
            outputs_D, _ = net(images,False,data_anchors[i:i+5,:,:])
            print('after net feeding')
            #handle NaN:
            # if (torch.norm((outputs_D != outputs_D).float())==0):
            #     if (i%50==0 or i%50==1):
            #         imshow_np(np.transpose(images[0,:,:,:].numpy(),(1,2,0)))
            #         imshow_np(np.transpose(outputs_D[0,:,:,:].detach().numpy(),(1,2,0)))
            loss = criterion(outputs_D, labels_D)
            loss.backward()
            optimizer.step()

            # compute statistics
            total += labels_D.size(0)
            running_loss += loss.item()
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / total))
        print('Epoch finished')
    print('Finished Training')


def prepare_dataloader_D(data_images_train, data_images_test, data_labels_D_train, data_labels_D_test):
    trainset_D = torch.utils.data.TensorDataset(torch.tensor(np.transpose(data_images_train, (0, 3, 1, 2))),
                                                torch.tensor(data_labels_D_train))
    testset_D = torch.utils.data.TensorDataset(torch.tensor(np.transpose(data_images_test, (0, 3, 1, 2))),
                                               torch.tensor(data_labels_D_test))

    trainloader_D = torch.utils.data.DataLoader(trainset_D, batch_size=5, shuffle=False)
    testloader_D = torch.utils.data.DataLoader(testset_D, batch_size=5, shuffle=False)

    return trainloader_D, testloader_D


if __name__ == "__main__":
    # creation des donnees:

    # Images
    Images = np.load('./Data/images.npz')

    # Changement de base
    Anchors = np.load('./Data/anchors.npz')

    # label_D:
    Labels_D = np.load('./Data/labels_D.npz')

    # label_spoofing:
    Labels = np.load('./Data/label.npz')

    # set:
    n = len(Images)

    data_images = np.zeros((n, 256, 256, 3), dtype=np.float32)
    data_anchors = np.zeros((n, 2, 4096), dtype=np.float32)
    data_labels_D = np.zeros((n, 32, 32, 1), dtype=np.float32)
    data_labels = np.zeros((n), dtype=np.float32)

    for item in Images.files:
        data_images[int(item), :, :, :] = Images[item]
        data_anchors[int(item), :, :] = Anchors[item]
        data_labels_D[int(item), :, :, :] = Labels_D[item]
        data_labels[int(item)] = Labels[item]

    training_part = 45 / 55
    n_train = int(n * training_part)

    # Training set
    data_images_train = data_images[:n_train, :, :, :]
    data_anchors_train = data_anchors[:n_train, :, :]
    data_labels_D_train = data_labels_D[:n_train, :, :, :]
    data_labels_train = data_labels[:n_train]

    # Test set
    data_images_test = data_images[n_train:, :, :, :]
    data_anchors_test = data_anchors[n_train:, :, :]
    data_labels_D_test = data_labels_D[n_train:, :, :, :]
    data_labels_test = data_labels[n_train:]


    trainloader_D, testloader_D = prepare_dataloader_D(data_images_train, data_images_test, data_labels_D_train,
                                                       data_labels_D_test)

    mon_model = Anti_spoof_net()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(mon_model.parameters(), lr=3e-3, betas=(0.9, 0.999), eps=1e-08)

    train_CNN(mon_model, optimizer, trainloader_D, data_anchors_train, criterion, n_epoch=1)