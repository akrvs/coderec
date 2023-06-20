import torch.nn as nn
import torch.optim as optim
from architectures.mlp_model import MlpModel
from embeddings.extraction import get_embeddings
from data.database import read_database
from training.train_loop import train_model
from training.plot_loss import plot_losses
from utils.preproccessing import *
from utils.preproccessing import data_loading
from architectures.lstm_model import LstmModel

# Global Variables
n_epochs = 100
batch_size = 128
window_length = 1

# Read the database.db
word_list = read_database()

# Preprocess data
word_list, word_to_int, int_to_word, n_words = preprocess_data(word_list)
WordModel = LstmModel(n_words)
embedding_model = nn.Sequential(WordModel.lstm)
X, y = create_sequences(word_list, word_to_int, window_length)
train_loader, val_loader = data_loading(X, y, test_size=0.2, batch_size=batch_size)

# Define the MLP model and it's parameters
embeddings_tensor = get_embeddings(embedding_model, train_loader)
MlpModel = MlpModel(embeddings_tensor.shape[1], 512, n_words)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MlpModel.to(device)
optimizer = optim.Adam(MlpModel.parameters(), weight_decay=1e-4)

train_loader, val_loader = data_loading(embeddings_tensor, y[:embeddings_tensor.shape[0]], test_size=0.2,
                                        batch_size=batch_size)
# Training
best_model, train_losses, val_losses = train_model(MlpModel, "mlp", train_loader, val_loader, optimizer,
                                                   n_epochs=n_epochs, loss_fn=None, word_list=word_list)

# Plot losses
plot_losses(train_losses, val_losses, n_epochs=n_epochs)

# Save the best model
torch.save(best_model, "mlp_model.pth")


