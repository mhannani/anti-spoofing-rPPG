import torch
import numpy as np


def sample_label(label: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Sample the given torch tensor and returns a tensor of zeros or ones

    :param label: torch.Tensor
        A tensor holding label of 5 images, [0., 0., 0., 0., 0.] for spoof or [1., 1., 1., 1., 1.] for real

    :param device: torch.Tensor
        Torch device for the returned tensor

    :return torch.Tensor
        A tensor holding labels of shape [5, 1, 2]
    """

    if torch.count_nonzero(label) == 0:
        return torch.zeros((5, 1, 2, 2), dtype=torch.float32).to(device)

    return torch.ones((5, 1, 2, 2), dtype=torch.float32).to(device)


if __name__ == "__main__":
    ex_label = torch.tensor(np.array([0, 0, 0, 0, 1]))
    print(sample_label(ex_label, device=torch.device('cpu')))
