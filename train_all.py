from utils.train import train_all
from utils.read import get_device, read_config_yaml


if __name__ == "__main__":
    # DONE: Implement function to train the two models in the same loop/time

    # read the configuration file
    cfg = read_config_yaml('./config.yaml')
    device = get_device(cfg)
    print('device: ', device)

    train_all(device=device)
