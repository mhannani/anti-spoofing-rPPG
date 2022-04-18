from torch.nn import Module
from torch import load
import os
import glob


def load_checkpoints(model_path: str) -> Module:
    """
    Load previously saved model
    :param model_path: str
        Absolute pretrained model path
    :parma last: bool
        True, loads the last checkpoint saved during last training process, False otherwise

    :return: torch.nn.Module
    """

    return load(model_path)


def check_saved_checkpoints(dir_path: str) -> bool:
    """
    Check for already saved checkpoints and load them.
    :param dir_path: str
        The director path to look for saved checkpoints
    :return bool
        True if a checkpoint was found, False otherwise
    """

    # check for pretrained existence
    if not os.path.exists(dir_path):
        print('No folder found')
        return False

    # list all checkpoints
    checkpoints = sorted(glob.glob(f'{dir_path}/*.pt'))

    if len(checkpoints) != 0:
        return True


def load_last_checkpoints(dir_path) -> Module:
    """
    Check and load the last checkpoints
    :param dir_path: str
        The directory path where checkpoint being saved
    :return torch.nn.Module
        A model serialized
    """

    if not check_saved_checkpoints(dir_path):
        raise ValueError('No checkpoints found in directory...')

    # list all checkpoints
    checkpoint_path = sorted(glob.glob(f'{dir_path}/*.pt'))[-1]

    return load_checkpoints(checkpoint_path)


if __name__ == "__main__":
    load_last_checkpoints('./pretrained')


