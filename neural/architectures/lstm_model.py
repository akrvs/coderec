import torch.nn as nn
import torch

'''class LstmModel(nn.Module):
    def __init__(self, n_words):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_words, hidden_size=16, num_layers=2, batch_first=True, dropout=0.2)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(16, n_words)

    def forward(self, x):
        """
        Performs the forward pass of the network.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size).

        Returns:
            Output tensor of shape (batch_size, output_size).
        """
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.linear(self.dropout(x))
        return x'''

class LstmModel(nn.Module):
    def __init__(self, n_class):
        super(LstmModel, self).__init__()
        self.lstm = nn.LSTM(input_size=n_class, hidden_size=16)
        self.W = nn.Linear(16, n_class, bias=False)
        self.b = nn.Parameter(torch.ones([n_class]))

    def forward(self, X):
        input = X.transpose(0, 1)
        hidden_state = torch.zeros(1, len(X), 16)
        cell_state = torch.zeros(1, len(X), 16)
        outputs, (_, _) = self.lstm(input, (hidden_state, cell_state))
        outputs = outputs[-1]
        model = self.W(outputs) + self.b
        return model
