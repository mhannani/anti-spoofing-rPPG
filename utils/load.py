from torch.nn import Module
from torch import load


def load_checkpoints(model_path: str) -> Module:
    """
    Load previously saved model

    :param model_path: str
        Abosulate pretrained model path
    :return: torch.nn.Module
    """

    return load(model_path)
