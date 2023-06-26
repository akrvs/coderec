import torch.nn as nn
import torch.optim as optim
from neural.architectures.mlp_model import MlpModel
from neural.embeddings.extraction import *
from data.database import read_database
from neural.train.train_loop import train_model
from neural.train import plot_losses
from utils.preproccessing import *
from utils.preproccessing import data_loading
from utils.load_or_train import load_or_train_lstm_model
from neural.lstm_predicting import generate_candidates

def train_mlp_model(database_path, window_length, batch_size=128, n_epochs=100):
    """
    Trains an MLP model using the provided database.

    Args:
        database_path: Path to the database file.
        window_length: Length of the input sequences.
        batch_size: Batch size for training. Default is 128.
        n_epochs: Number of training epochs. Default is 100.

    Returns:
        The trained MLP model.
    """
    word_list = read_database(database_path)

    word_list, word_to_int, int_to_word, n_words = preprocess_data(word_list)

    pretrained_model_path = "/Users/akrvs/PycharmProjects/Project/lstm_model.pth"
    best_lstm_model = load_or_train_lstm_model(database_path, window_length, batch_size, n_epochs,
                                               pretrained_model_path,
                                               n_words)

    print()
    candidates = generate_candidates(initial_prompt=input("Enter a word from the database: "),
                                     window_length=window_length, n_words=n_words, word_to_int=word_to_int,
                                     int_to_word=int_to_word, best_lstm_model=best_lstm_model)

    embedding_model = nn.Sequential(best_lstm_model.lstm)
    X, y = create_sequences(word_list, word_to_int, window_length)

    candidate_embeddings = get_candidate_embeddings(embedding_model=embedding_model, candidates=candidates,
                                                    window_length=window_length, n_words=n_words,
                                                    word_to_int=word_to_int)

    mlp_model = MlpModel(candidate_embeddings.shape[1], 512, n_words)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mlp_model.to(device)
    optimizer = optim.Adam(mlp_model.parameters(), weight_decay=1e-4)

    train_loader, val_loader = data_loading(candidate_embeddings, y[:candidate_embeddings.shape[0]],
                                            test_size=0.2, batch_size=batch_size)

    best_model, train_losses, val_losses, similarity_results = train_model(mlp_model, "mlp", train_loader, val_loader,
                                                                           optimizer, n_epochs=n_epochs, loss_fn=None,
                                                                           word_list=word_list)

    plot_losses(train_losses, val_losses, n_epochs=n_epochs)

    torch.save(best_model, "/Users/akrvs/PycharmProjects/Project/mlp_model.pth")

    return best_model, similarity_results, candidates
