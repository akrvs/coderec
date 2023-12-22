import torch
import os
import neural

def load_or_train_lstm_model(model_class, database, word_list, word_to_int, int_to_word, window_length, batch_size, n_epochs,
                             pretrained_model_path, n_words):
    """
        Loads a pre-trained LSTM model if available, otherwise trains a new model.

        Args:
            model_class: The class of the LSTM model to train.
            database_path: Path to the database.
            window_length: Length of the input window.
            batch_size: Batch size for training.
            n_epochs: Number of epochs to train.
            pretrained_model_path: Path to the pre-trained model.
            n_words: Number of words in the vocabulary.

        Returns:
            The loaded or trained LSTM model.
        """

    if os.path.exists(pretrained_model_path):
        print("Pre-trained model found. Loading the model...")
        best_lstm_model = model_class(n_words)
        torch.load("/Users/akrvs/PycharmProjects/Project/lstm_model.pth")
        torch.load("/Users/akrvs/PycharmProjects/Project/word_to_int.pth")
        torch.load("/Users/akrvs/PycharmProjects/Project/int_to_word.pth")
        best_lstm_model.eval()

        return best_lstm_model, word_to_int, int_to_word

    else:
        print("Pre-trained model not found. Training a new model...")
        best_lstm_model, word_to_int, int_to_word = neural.train_lstm_model(model_class, word_list, word_to_int,
                                                                               int_to_word, n_words=n_words, database=database,
                                                                               window_length=window_length,
                                                                               batch_size=batch_size, n_epochs=n_epochs)
        torch.save(best_lstm_model, pretrained_model_path)
        torch.save(word_to_int, "/Users/akrvs/PycharmProjects/Project/word_to_int.pth")
        torch.save(int_to_word, "/Users/akrvs/PycharmProjects/Project/int_to_word.pth")

        return best_lstm_model, word_to_int, int_to_word