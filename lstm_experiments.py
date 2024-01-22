from data.database import open_json_file
from data.preproccessing import *
from flask_app.run_experiments import run_experiment
from utils import lukasiewicz_implication_2
from utils.average_cosine_similarity import calculate_average_cosine_similarity

# Global Variables
batch_size = 128
learning_rate = 0.01
num_genes = 100
num_epochs = 200
window_length = 3

# Paths for the LSTM model
pretrained_model_path = "/Users/akrvs/PycharmProjects/Project/lstm_model.pth"
file_path = '/Users/akrvs/PycharmProjects/Project/web-train.json'

# Read the database
data = open_json_file(file_path)

# Some preprocessing
sentences = [remove_question_marks(entry["Question"].lower()) for entry in data["Data"]]
sentences = sentences[1:3500]
max_len = max(len(sentence.split()) for sentence in sentences)
sentences = [pad_sequence(sentence, max_len) for sentence in sentences]

# User's prompt
prompt = input("Enter prompt here: ")

# Run experiment
results, selected_indices, selected_embeddings, MLP_output, candidates, candidate_embeddings = run_experiment(prompt,
                        model_architecture='LSTM', num_epochs=num_epochs, batch_size=batch_size,
                         learning_rate=learning_rate, num_genes=num_genes, loss_function=lukasiewicz_implication_2,
                        window_length=window_length, database=sentences, pretrained_model_path=pretrained_model_path)

# Calculate cosine similarity and print the results
print()
print("Most Dissimilar Sentences:")
for i, indices in enumerate(selected_indices):
    print(f"{i + 1}. {[candidates[idx] for idx in indices]}")

average_cosine_similarity = calculate_average_cosine_similarity(candidate_embeddings)
print("Average cosine similarity for LSTM:", average_cosine_similarity)

average_cosine_similarity = calculate_average_cosine_similarity(MLP_output)
print("Average cosine similarity after the MLP:", average_cosine_similarity)

average_cosine_similarity = calculate_average_cosine_similarity(selected_embeddings)
print("Average cosine similarity after the Genetic Algorithm:", average_cosine_similarity)