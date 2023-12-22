import torch.nn as nn
import torch.optim as optim
import neural
import torch

def train_lstm_model(model_class, word_list, word_to_int, int_to_word, n_words, database=None, window_length=3, batch_size=128,
                     n_epochs=100):
    """
        Trains an LSTM model using the provided database.

        Args:
            model_class: The class of the LSTM model to train.
            database_path: Path to the database file. Default is None.
            window_length: Length of the input sequences. Default is 3.
            batch_size: Batch size for training. Default is 128.
            n_epochs: Number of training epochs. Default is 100.

        Returns:
            The trained LSTM model.
        """

    WordModel = model_class(n_words)
    optimizer = optim.Adam(WordModel.parameters(), weight_decay=5.E-4)
    loss_fn = nn.CrossEntropyLoss()

    best_model, word_to_int, int_to_word, train_losses, accuracy_list = neural.train_model(
        WordModel, "lstm", optimizer, n_epochs, word_to_int, int_to_word, n_words, database, train_loader=None,
        val_loader=None, loss_fn=loss_fn, word_list=word_list)

    neural.plot_losses(train_losses, accuracy_list, n_epochs=n_epochs, model_type="lstm")

    return best_model, word_to_int, int_to_word


