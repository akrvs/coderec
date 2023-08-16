from data.database import read_database
from neural.mlp_training import train_mlp_model
from utils.load_or_train import load_or_train_lstm_model
from cluster_analysis.genetic import genetic_algorithm
from utils.lstm_predicting import generate_candidates
import neural
import data
import utils

# Global Variables
n_epochs = 50
batch_size = 128
window_length = 1
k = 5
num_genes = 10
num_permutations = 10
num_epochs = 30

# Loading models
pretrained_model_path = "/Users/akrvs/PycharmProjects/Project/lstm_model.pth"
database_path = '/Users/akrvs/PycharmProjects/Project/Isaac_Asimov.txt'
word_list = read_database(database_path)
word_list, word_to_int, int_to_word, n_words = data.preprocess_data(word_list)
best_lstm_model = load_or_train_lstm_model(neural.LstmModel, database_path, window_length, batch_size, n_epochs,
                                           pretrained_model_path, n_words)
# Loading the candidates
print()
candidates = generate_candidates(initial_prompt=input("Enter a word from the database: "),
                                       window_length=window_length, n_words=n_words, word_to_int=word_to_int,
                                       int_to_word=int_to_word, best_lstm_model=best_lstm_model)
best_mlp_model = train_mlp_model(neural.MlpModel, database_path, best_lstm_model,
                                 candidates, window_length, batch_size, n_epochs)

# Predictions
candidate_embeddings = neural.get_candidate_embeddings(embedding_model=best_lstm_model.lstm, candidates=candidates,
                                                       window_length=window_length, n_words=n_words,
                                                       word_to_int=word_to_int)

similarity_results = utils.cosine_similarity(candidate_embeddings, candidate_embeddings)

for i, candidate in enumerate(candidates):
    print(f"\nCosine Similarity results for candidate {candidate}:")
    for j, other_candidate in enumerate(candidates):
        similarity = similarity_results[j][i]
        print(f"Cosine Similarity with candidate {other_candidate}: {similarity:.4f}")

results = genetic_algorithm(candidates, similarity_results, k, num_genes, num_permutations, num_epochs)
print(results)