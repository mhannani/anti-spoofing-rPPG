from torch.nn import Module
from torch import load
import os


def load_checkpoints(model_path: str) -> Module:
    """
    Load previously saved model

    :param model_path: str
        Absolute pretrained model path
    :return: torch.nn.Module
    """

    return load(model_path)


def check_saved_checkpoints(epoch: int) -> bool:
    """
    Check for already saved checkpoints and load them.
    """

    # check for pretrained existence
    if os.path.exists('./pretrained'):
        return False

    # list all checkpoints
    os.listdir('./pretrained/')
