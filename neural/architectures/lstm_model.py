import torch.nn as nn

class LstmModel(nn.Module):
    def __init__(self, n_words):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_words, hidden_size=256, num_layers=2, batch_first=True, dropout=0.2)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(256, n_words)

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
        return x
