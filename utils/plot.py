import os
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from utils.NPZ_loader import NPZLoader
from utils.split import get_sets
import torch
from torchvision.utils import save_image


def plot_ndarray(array: np.ndarray) -> None:
    """
    Displays image from numpy 's ndarray
    """

    img = Image.fromarray(array.astype('uint8')).convert('RGBA')
    img.save('images/img_1.png')


def show_depth_maps(batch: torch.Tensor) -> None:
    """
    Shows the depth map of each image in the batch.
    """

    # check if the diagnosis directory exists, otherwise create it
    if not os.path.exists('diagnosis'):
        os.mkdir('diagnosis')

    # unpack the batch
    _images, label_d, anchors, label = batch
    label_index = torch.all(label.bool()).item()

    print('anchors:', anchors.shape)
    # saving images
    for i, image in enumerate(_images):
        save_image(image, f'img_{i}_class_{label_index}.png')

    _images = torch.squeeze(_images)
    print(_images.shape)


if __name__ == "__main__":
    plot_ndarray(np.random.rand(256, 256, 3) * 255)
    # print(np.random.randint(0, 1, size=(256, 256, 3), dtype=np.float64))

    # Get data
    dataset = NPZLoader('./Data')

    # split dataset
    train_set, _ = get_sets(dataset, 0.5)

    # Get train dataloader
    train_data = DataLoader(train_set, batch_size=5)

    for i, data in enumerate(train_data):
        images, _, _, _ = data
        print(images.shape)
        break



