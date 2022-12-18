#! python3

__author__ = "Simanta Barman"
__email__  = "barma017@umn.edu"

import torch
from torch import nn


class RNN_0(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, n_categories, dropout):
        super(RNN_0, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        num_classes = input_size
        self.i2h = nn.Linear(n_categories + input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(n_categories + input_size + hidden_size, num_classes)
        self.o2o = nn.Linear(hidden_size + num_classes, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, x, hidden):
        combined = torch.cat((category, x, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self, x_size):
        return torch.zeros(1, self.hidden_size)


class RNN_1(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, n_categories, dropout):
        super(RNN_1, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        num_classes = input_size
        
        self.rnn = nn.RNN(input_size + n_categories, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

        self.o2o = nn.Linear(hidden_size + num_classes, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, x, hidden):
        combo = torch.concat((category, x), 1)
        out, hidden = self.rnn(combo, hidden)
        out = self.fc(out)

        output_combined = torch.cat((hidden, out), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self, x_size):
        return torch.zeros(self.num_layers, x_size, self.hidden_size).squeeze(1)


class RNN_GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, n_categories, dropout):
        super(RNN_GRU, self).__init__()

        num_classes = input_size

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.gru = nn.GRU(input_size + n_categories, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.o2o = nn.Linear(hidden_size + num_classes, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, x, hidden):
        combo = torch.concat((category, x), 1)
        out, hidden = self.gru(combo, hidden)

        out = self.fc(out)
        output_combined = torch.cat((hidden, out), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self, x_size):
        return torch.zeros(self.num_layers, x_size, self.hidden_size).squeeze(1)


class RNN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, n_categories, dropout):
        super(RNN_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        num_classes = input_size
        self.lstm = nn.LSTM(input_size + n_categories, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.o2o = nn.Linear(hidden_size + num_classes, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, x, hidden):
        combo = torch.concat((category, x), 1)
        # h0, c0 = hidden
        h0 = torch.zeros(self.num_layers, 1, self.hidden_size).squeeze(1)
        c0 = torch.zeros(self.num_layers, 1, self.hidden_size).squeeze(1)
        
        # Forward propagate LSTM
        out, hidden = self.lstm(combo, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        h0 = hidden[0]
        # out = out.reshape(out.shape[0], -1)

        out = self.fc(out)
        output_combined = torch.cat((h0, out), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden 

    def initHidden(self, x_size):
        return None
    
