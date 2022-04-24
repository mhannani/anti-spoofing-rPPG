import torch.nn
from torch import device


def train_on(net: str, device: device) -> None:
    """
    Shows which net is training on which device
    :param net: str
        which network is training 'cnn', 'rnn' for CNN and RNN models respectively
    :param device: torch.device
        Torch device to train on.

    :return: None
    """

    if device == 'cpu':
        print(f'Training {net} on cpu...')
    else:
        torch.cuda.empty_cache()
        print(f'Training {net} on gpu...')

    if net not in ['cnn', 'rnn']:
        raise NotImplemented('Net value is not implemented !')
