from torch.nn import Module
from torch import save
import os


def save_checkpoints(model: Module, epoch: int) -> None:
    """
    Save checkpoints with the provided filename
    :param model: torch.nn.Module
        PyTorch model
    :param epoch: int
        current epoch
    :return: None
    """

    # check if the pretrained folder exists
    if not os.path.exists('pretrained'):
        os.mkdir('pretrained')

    # save the model
    print(f'saving model to pretrained/model_epoch_{epoch}.pt file')
    save(model, f'pretrained/model_epoch_{epoch}.pt')
