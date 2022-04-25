import torch
from utils.load import check_saved_checkpoints
from utils.load import load_last_checkpoints
from models.Cnn_Rnn import CnnRnn


def create_network(resume_training: bool, device: torch.device):
    """
    Create/load checkpoint network on provided device

    :param resume_training: bool
        True to resume training on previously saved checkpoint
    :param device: torch.device
        The torch device to train on
    """

    if resume_training and check_saved_checkpoints('./pretrained'):
        print('Loading saved model and resume training...')
        model = load_last_checkpoints('./pretrained/').to(device)
    else:
        print('No checkpoint to resume training from... Training from scratch.')
        model = CnnRnn(device)
        model = model.to(device)

    return model
