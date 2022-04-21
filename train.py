from utils.train import train
from utils.read import get_device, read_config_yaml


if __name__ == "__main__":
    # TODO: Implement function to train the two models in the same loop/time

    # read the configuration file
    cfg = read_config_yaml('./config.yaml')
    device = get_device(cfg)

    train(device=device, net='cnn')
    # train(net='rnn')
