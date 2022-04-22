from utils.load import load_last_checkpoints


def inference(batch, epoch: int) -> float:
    """
    Doing inference with CNN_RNN network.
    :param: batch
        Batch of images
    """

    # load pretrained model
    model = load_last_checkpoints('./pretrained')
    depth_map, rppg = model(None, turned=True, anchors=None)
    pass


