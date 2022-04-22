import torch
import yaml
from typing import Dict


def read_config_yaml(config_path: str) -> Dict:
    """
    Reads the configuration file
    :param config_path: str
        The absolute path to the configuration file
    :return Dict
        Config file as python object
    """

    with open(config_path, 'r') as config:
        config_file = yaml.safe_load(config)
        return config_file


def get_device(config: Dict) -> torch.device:
    """
    Get the device based on configuration
    :param config: Dict
        Python object containing the configuration
    :return torch.device
        The device selected in configuration
    """

    device = None
    config_device = config['device']

    if config_device == " ":
        device = torch.device("cpu")
    elif config_device == "0":
        device = torch.device('cuda:0')
    elif config_device == "1":
        device = torch.device("cuda:1")
    else:
        raise NotImplementedError

    return device


if __name__ == "__main__":
    config_dict = read_config_yaml('config.yaml')
    get_device(config_dict)
