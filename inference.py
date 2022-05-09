import torch
from torch.utils.data import DataLoader
from utils.inference import inference
from utils.NPZ_loader import NPZLoader
from utils.split import get_sets
from utils.plot import show_depth_maps

if __name__ == "__main__":
    # Get data
    dataset = NPZLoader('./Data')

    # split dataset
    _, test_set = get_sets(dataset, 0.8)

    # Get train dataloader
    train_data = DataLoader(test_set, batch_size=5)

    images, label_d, anchors, label = next(iter(train_data))

    for i, batch in enumerate(train_data):
        images, label_d, anchor, label = batch

        # show images in batch
        # show_depth_maps(batch)

        # inference
        inference(images, 12)


