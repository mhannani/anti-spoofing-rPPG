from utils.train import train
from utils.load import load_last_checkpoints


if __name__ == "__main__":
    # TODO: Implement function to train the two models in the same loop/time

    train(net='cnn')
    train(net='rnn')
