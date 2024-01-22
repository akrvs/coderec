import torch.nn as nn
import torch

class LstmModel(nn.Module):
    def __init__(self, n_class):
        super(LstmModel, self).__init__()
        self.lstm = nn.LSTM(input_size=n_class, hidden_size=16)
        self.W = nn.Linear(16, n_class, bias=False)
        self.b = nn.Parameter(torch.ones([n_class]))

    def forward(self, X):
        """
        Forward pass of the model.

        Args:
            X (torch.Tensor): Input tensor with shape (batch_size, sequence_length, input_size).

        Returns:
            torch.Tensor: Output tensor after the forward pass.
        """
        input_ = X.transpose(0, 1)
        hidden_state = torch.zeros(1, len(X), 16)
        cell_state = torch.zeros(1, len(X), 16)
        outputs, (_, _) = self.lstm(input_, (hidden_state, cell_state))
        outputs = outputs[-1]
        model = self.W(outputs) + self.b
        return model
