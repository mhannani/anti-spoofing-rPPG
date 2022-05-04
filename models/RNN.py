import torch
import torch.nn as nn
import numpy as np
import torchvision
import torch.fft


class RNN(nn.Module):

    def __init__(self, device):
        """
        RNN's class constructor
        """
        # superclass' constructor
        super().__init__()

        # parameters.
        self.hidden_dim = 100
        self.input_dim = 32 * 32
        self.num_layers = 1
        self.batch_size = 5

        self.hidden = (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim, device=self.device),
                       torch.zeros(self.num_layers, self.batch_size, self.hidden_dim, device=self.device))
        # LSTM cell
        self.LSTM = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers)

        # fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 2)

    def forward(self, f):
        """
        Forward pass
        """

        # F est de dimension [5,32,32,1]
        f = f.view(5, 1, -1)

        output, (hidden, _) = self.LSTM(f, self.hidden)

        R = self.fc(output)
        R = torch.fft.fft(R, norm='backward', dim=1)

        R = torch.view_as_real(R)

        return R # F[5,1,2]
