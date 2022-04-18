from utils.train import train


if __name__ == "__main__":
    # TODO: Implement function to train the two models in the same loop/time

    train(net='cnn')
    train(net='rnn')
