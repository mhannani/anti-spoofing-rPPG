import torch
from utils.load import load_last_checkpoints


def inference(batch, lambda_value: int = 10):
    """
    Doing inference with CNN_RNN network.
    :param: batch
        Batch of images
    """

    # load pretrained model
    model = load_last_checkpoints('./pretrained')
    depth_map, rppg = model(batch, turned=False, anchors=None)
    print(torch.pow(torch.abs(depth_map), 2))

    # return depth_map.shape, rppg.shape
