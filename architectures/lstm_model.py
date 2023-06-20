import torch.nn as nn

class LstmModel(nn.Module):
    def __init__(self, n_words):
        print("Lstm Model is currently up!")
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_words, hidden_size=256, num_layers=2, batch_first=True, dropout=0.2)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(256, n_words)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.linear(self.dropout(x))
        return x