from data.database import read_database
from utils.preproccessing import *
from neural.mlp_training import train_mlp_model
import torch
from utils.load_or_train import load_or_train_lstm_model

n_epochs = 100
batch_size = 128
window_length = 1

pretrained_model_path = "/Users/akrvs/PycharmProjects/Project/lstm_model.pth"
database_path = '/Users/akrvs/PycharmProjects/database.db'
word_list = read_database(database_path)

word_list, word_to_int, int_to_word, n_words = preprocess_data(word_list)

best_mlp_model, similarity_results, candidates = train_mlp_model(database_path, window_length, batch_size, n_epochs)

similarity_results = torch.cat(similarity_results)

best_lstm_model = load_or_train_lstm_model(database_path, window_length, batch_size, n_epochs,
                                           pretrained_model_path, n_words)

for i, candidate in enumerate(candidates):
    print(f"\nCosine Similarity results for candidate {candidate}:")
    for j, other_candidate in enumerate(candidates):
        similarity = similarity_results[j][i]
        print(f"Cosine Similarity with candidate {other_candidate}: {similarity}")




