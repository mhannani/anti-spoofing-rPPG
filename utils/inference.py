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
    score = torch.sqrt(torch.norm(rppg)) + 1.0 * torch.sqrt(torch.norm(depth_map))

    print('score: ', score)
    # return depth_map.shape, rppg.shape
