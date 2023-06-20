import torch.nn as nn
import torch.optim as optim
from architectures.lstm_model import LstmModel
from data.database import read_database
from training.train_loop import train_model
from training.plot_loss import plot_losses
from utils.preproccessing import *

# Global Variables
n_epochs = 100
window_length = 1
batch_size = 128

# Read the database.db
word_list = read_database()

# Preprocess data
word_list, word_to_int, int_to_word, n_words = preprocess_data(word_list)
X, y = create_sequences(word_list, word_to_int, window_length)
train_loader, val_loader = data_loading(X, y, test_size=0.2, batch_size=batch_size)

# Define the LSTM model and it's parameters
WordModel = LstmModel(n_words)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
WordModel.to(device)
optimizer = optim.Adam(WordModel.parameters())
loss_fn = nn.CrossEntropyLoss()

# Training
best_model, train_losses, val_losses = train_model(WordModel, "lstm", train_loader, val_loader, optimizer,
                                                   n_epochs=n_epochs, loss_fn=loss_fn, word_list=None)

# Plot losses
plot_losses(train_losses, val_losses, n_epochs=n_epochs)

# Save the best model
torch.save(best_model, "lstm_model.pth")


