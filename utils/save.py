from torch.nn import Module
from torch import save


def save_checkpoints(model: Module, epoch: int) -> None:
    """
    Save checkpoints with the provided filename
    :param model: torch.nn.Module
        PyTorch model
    :param epoch: int
        current epoch
    :return: None
    """

    save(model, f'model_epoch_{epoch}')

