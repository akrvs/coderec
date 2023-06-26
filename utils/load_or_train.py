import torch
import os
from neural.architectures.lstm_model import LstmModel
from neural.lstm_training import train_lstm_model

def load_or_train_lstm_model(database_path, window_length, batch_size, n_epochs, pretrained_model_path, n_words):
    """
        Loads a pre-trained LSTM model if available, otherwise trains a new model.

        Args:
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
        best_lstm_model = LstmModel(n_words)
        best_lstm_model.load_state_dict(torch.load(pretrained_model_path))
        best_lstm_model.eval()
    else:
        print("Pre-trained model not found. Training a new model...")
        trained_lstm_model = train_lstm_model(database_path=database_path, window_length=window_length,
                                              batch_size=batch_size, n_epochs=n_epochs)
        best_lstm_model = LstmModel(n_words)
        best_lstm_model.load_state_dict(torch.load(pretrained_model_path))
        best_lstm_model.eval()

    return best_lstm_model

