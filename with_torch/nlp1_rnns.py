#! python3

__author__ = "Simanta Barman"
__email__  = "barma017@umn.edu"

import torch
from torch import nn


class RNN_0(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN_0, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, hidden):
        combined = torch.cat((x, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self, x_size):
        return torch.zeros(1, self.hidden_size)


class RNN_1(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN_1, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        # out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        out = self.softmax(out)
        return out, hidden

    def initHidden(self, x_size):
        return torch.zeros(self.num_layers, x_size, self.hidden_size).squeeze(1)


class RNN_GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN_GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, hidden):
        out, hidden = self.gru(x, hidden)
        # out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        out = self.softmax(out)
        return out, hidden

    def initHidden(self, x_size):
        return torch.zeros(self.num_layers, x_size, self.hidden_size).squeeze(1)


class RNN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, hidden):
        # h0, c0 = hidden
        h0 = torch.zeros(self.num_layers, 1, self.hidden_size).squeeze(1)
        c0 = torch.zeros(self.num_layers, 1, self.hidden_size).squeeze(1)
        # Forward propagate LSTM
        out, *hidden = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        # out = out.reshape(out.shape[0], -1)

        # Decode the hidden state of the last time step
        out = self.fc(out)
        out = self.softmax(out)
        return out, hidden

    def initHidden(self, x_size):
        return (torch.zeros(self.num_layers, 1, self.hidden_size).squeeze(1), 
                torch.zeros(self.num_layers, 1, self.hidden_size).squeeze(1))
    
