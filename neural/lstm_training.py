import torch.nn as nn
import torch.optim as optim
import neural
import torch

def train_lstm_model(model_class, word_list, word_to_int, int_to_word, n_words, database=None, window_length=3, batch_size=128,
                     n_epochs=100):
    """
        Trains an LSTM model using the specified parameters.

        Args:
            model_class: The class of the LSTM model.
            word_list (list): List of unique words.
            word_to_int (dict): A dictionary mapping words to their corresponding indices.
            int_to_word (dict): A dictionary mapping indices to their corresponding words.
            n_words (int): The total number of unique words.
            database (str): Path to the database file (optional).
            window_length (int): Length of the input sequence window (default: 3).
            batch_size (int): Size of each training batch (default: 128).
            n_epochs (int): The number of training epochs (default: 100).

        Returns:
            tuple: A tuple containing:
                - best_model (torch.nn.Module): The best-trained LSTM model.
                - word_to_int (dict): Updated word_to_int dictionary.
                - int_to_word (dict): Updated int_to_word dictionary.
    """

    WordModel = model_class(n_words)
    optimizer = optim.Adam(WordModel.parameters(), weight_decay=5.E-4)
    loss_fn = nn.CrossEntropyLoss()

    best_model, word_to_int, int_to_word, train_losses, accuracy_list = neural.train_model(
        WordModel, "lstm", optimizer, n_epochs, word_to_int, int_to_word, n_words, database, train_loader=None,
        val_loader=None, loss_fn=loss_fn, word_list=word_list)

    neural.plot_losses(train_losses, accuracy_list, n_epochs=n_epochs, model_type="lstm")

    return best_model, word_to_int, int_to_word


