import matplotlib.pyplot as plt
import numpy
import torch


def imshow_np(img: numpy.ndarray):
    """
    Show the image provided as ndarray

    :param img: numpy.ndarray
    :return: None
    """

    # get the dimension
    h, w, d = img.shape

    # one depth/ gray image
    if d == 1:
        img = img[:, :, 0]

    # show the image
    plt.imshow(img)
    plt.show()


def convert_img(img: torch.Tensor):
    """
    Converts tensor to numpy

    :param img: torch.Tensor
        Image as tensor
    :return: numpy.ndarray
        Returned image representation as ndarray
    """

    return img.numpy()
