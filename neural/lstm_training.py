import torch.nn as nn
import torch.optim as optim
from data.database import read_database
from data.preproccessing import *
import neural

def train_lstm_model(model_class, database_path=None, window_length=1, batch_size=128, n_epochs=100):
    """
        Trains an LSTM model using the provided database.

        Args:
            model_class: The class of the LSTM model to train.
            database_path: Path to the database file. Default is None.
            window_length: Length of the input sequences. Default is 1.
            batch_size: Batch size for training. Default is 128.
            n_epochs: Number of training epochs. Default is 100.

        Returns:
            The trained LSTM model.
        """
    if database_path is None:
        raise ValueError("Database path is required.")

    word_list = read_database(database_path)

    word_list, word_to_int, int_to_word, n_words = preprocess_data(word_list)
    X, y = create_sequences(word_list, word_to_int, window_length)
    train_loader, val_loader = data_loading(X, y, test_size=0.2, batch_size=batch_size)

    WordModel = model_class(n_words)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    WordModel.to(device)
    optimizer = optim.Adam(WordModel.parameters())
    loss_fn = nn.CrossEntropyLoss()

    best_model, train_losses, val_losses = neural.train_model(
        WordModel, "lstm", train_loader, val_loader, optimizer,
        n_epochs=n_epochs, loss_fn=loss_fn, word_list=None
    )

    neural.plot_losses(train_losses, val_losses, n_epochs=n_epochs)

    torch.save(best_model, "/Users/akrvs/PycharmProjects/Project/lstm_model.pth")

    return best_model


