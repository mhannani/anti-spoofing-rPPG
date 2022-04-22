#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 16:29:19 2022

@author: amine
"""

import sys
from torch.utils import data
import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import Anti_Spoof_net
from tqdm import tqdm

from utils.NPZ_loader import NPZLoader


# Display function
def imshow_np(img):
    height, width, depth = img.shape
    if depth == 1:
        img = img[:, :, 0]
    plt.imshow(img)
    plt.show()


def imshow(img):
    imshow_np(img.numpy())


# def prepare_dataloader_d(data_images_train, data_images_test, data_labels_D_train, data_labels_D_test):
#
#     trainset_D = data.TensorDataset(torch.tensor(np.transpose(data_images_train, (0, 3, 1, 2))), torch.tensor(data_labels_D_train))
#     # testset_D = data.TensorDataset(torch.tensor(np.transpose(data_images_test, (0, 3, 1, 2))), torch.tensor(data_labels_D_test))
#
#     trainloader_D = data.DataLoader(trainset_D,batch_size=5,shuffle=False,pin_memory=False)
#     # testloader_D = data.DataLoader(testset_D,batch_size=5,shuffle=False,pin_memory=False)
#
#     return trainloader_D


def load_data(set_name: str):
    """
    Returns data loader of the given setname.
    :param set_name: str
        The set name, 'train' for training set and 'test' for test set.
    :return: torch.utils.data.Datalaoder
    """

    if set_name not in ['train', 'test']:
        raise ValueError('Not a valid test')

    train_set_d = NPZLoader('./Data/')
    # Convert train data into tensors
    if set_name == 'train':
        # train_set_d = data.TensorDataset(torch.tensor(np.transpose(data_images_train, (0, 3, 1, 2))),
        #                                  torch.tensor(data_labels_D_train))
        return data.DataLoader(train_set_d, batch_size=5, shuffle=False, pin_memory=False, num_workers=0)

    # Convert test data into tensors
    # if set_name == 'test':
        # test_set_d = data.TensorDataset(torch.tensor(np.transpose(data_images_test, (0, 3, 1, 2))),
        #                                 torch.tensor(data_labels_D_test))
        # return data.DataLoader(test_set_d, batch_size=5, shuffle=False, pin_memory=False, num_workers=3)


# Training
def train_cnn(net, optimizer, trainloader, data_anchors, criterion, n_epoch=10):
    print('training cnn...')
    total = 0
    for epoch in tqdm(range(n_epoch)):
        # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader):

            # Pre-created Data:
            images, labels_D, _, _ = data

            print('shape information')
            print(images.shape)
            print(labels_D.shape)
            print(data_anchors[i:i + 5, :, :].shape)
            print('=================')

            images, labels_D = images, labels_D
            # training step
            optimizer.zero_grad()
            outputs_D, _ = net(images, False, data_anchors[i:i + 5, :, :])

            # handle NaN:
            if (torch.norm((outputs_D != outputs_D).float()) == 0):
                # if (i%50==0 or i%50==1):
                #     imshow_np(np.transpose(images[0,:,:,:].numpy(),(1,2,0)))
                #     imshow_np(np.transpose(outputs_D[0,:,:,:].detach().numpy(),(1,2,0)))

                loss = criterion(outputs_D, labels_D)
                loss.backward(retain_graph=True)
                optimizer.step()
                # compute statistics
                total += labels_D.size(0)
                running_loss += loss.item()
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / total))

        print('Epoch finished')
    print('Finished Training')


def train_RNN(net, optimizer, trainloader, anchors, labels, criterion, n_epoch=10):
    total = 0
    for epoch in range(n_epoch):
        # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # Donnees pre-crees:
            images, labels_D, _, _ = data
            # training step
            optimizer.zero_grad()
            _, outputs_F = net(images, False, anchors[i:i + 1, :, :])

            # handle NaN:
            if (torch.norm((outputs_F != outputs_F).float()) == 0):
                if labels[i * 5] == 0:  # toutes les images du batch proviennent de la même vidéo
                    label = torch.zeros((5, 1, 2), dtype=torch.float32)
                else:
                    label = torch.ones((5, 1, 2), dtype=torch.float32)

                loss = criterion(outputs_F, outputs_F)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(net.parameters(), max_norm=10)
                optimizer.step()

                # compute statistics
                total += labels_D.size(0)
                running_loss += loss.item()
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / total))
        print('Epoch finished')
    print('Finished Training')


# For the overall model training, we alternatively train the CNN part and the CNN/RNN part
def train_All(net, optimizer, trainloader, anchors, labels, criterion, n_epoch=1):
    for i in range(n_epoch):
        net = torch.load('mon_model')
        train_cnn(net, optimizer, trainloader, data_anchors_train, criterion, n_epoch=1)
        # torch.save(net, 'mon_model')
        # train_RNN(net, optimizer, trainloader, data_anchors_train, data_labels_train, criterion, n_epoch=1)
        # torch.save(net, 'mon_model')

if __name__ == "__main__":
    # Data sets creation
    # creation des donnees:

    # Images
    Images = np.load('./Data/images.npz')
    print(type(Images))

    # Changement de base
    Anchors = np.load('./Data/anchors.npz')

    # label_D:
    Labels_D = np.load('./Data/labels_D.npz')

    # label_spoofing:
    Labels = np.load('./Data/label.npz')

    # set:
    n = len(Images)
    # n=5

    data_images = np.zeros((n, 256, 256, 3), dtype=np.float32)
    data_anchors = np.zeros((n, 2, 4096), dtype=np.float32)
    data_labels_D = np.zeros((n, 32, 32, 1), dtype=np.float32)
    data_labels = np.zeros((n), dtype=np.float32)

    for item in Images.files:
        data_images[int(item), :, :, :] = Images[item]
        print('Images[item]==>: ', Images[item].shape)
        data_anchors[int(item), :, :] = Anchors[item]
        data_labels_D[int(item), :, :, :] = Labels_D[item]
        data_labels[int(item)] = Labels[item]

        print('data_images: ', data_images.shape)
        print('data_images: ', data_anchors.shape)
        print('data_images: ', data_labels_D.shape)
        print('data_images: ', data_labels)
        break

    # training_part = 1 / 100
    training_part = 1 / 200
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

    # trainloader_D = prepare_dataloader_d(data_images_train, data_images_test, data_labels_D_train,
    #                                                    data_labels_D_test)

    train_loader_d = load_data('train')

    # get train_set loader
    # Model creation

    print('instantiate the model')
    mon_model = Anti_Spoof_net.Anti_spoof_net()
    # mon_model = torch.load('mon_model')
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(mon_model.parameters(), lr=3e-3, betas=(0.9, 0.999), eps=1e-08)

    mon_model = torch.load('mon_model')

    print('start training...')
    train_All(mon_model, optimizer, train_loader_d, data_anchors_train, data_labels_train, criterion, n_epoch=10)
