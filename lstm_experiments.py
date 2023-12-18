import pandas as pd
import json
from flask_app.run_experiments import run_experiment
from utils import lukasiewicz_implication_2
from utils.average_cosine_similarity import calculate_average_cosine_similarity

# Global Variables
batch_size = 128
learning_rate = 0.01
num_genes = 100
num_epochs = 100
window_length = 5

# Paths for the LSTM model
pretrained_model_path = "/Users/akrvs/PycharmProjects/Project/lstm_model.pth"
file_path = '/Users/akrvs/PycharmProjects/Project/web-train.json'

# Load the JSON data from file
with open(file_path, 'r') as file:
    data = json.load(file)

lstm_data = [entry["Question"] for entry in data["Data"]]
lstm_data = lstm_data[:3500]
df_lstm = pd.DataFrame(lstm_data, columns=['Question'])
database_path = "/Users/akrvs/PycharmProjects/Project/lstm.csv"
df_lstm.to_csv(database_path, sep='\t', index=False)

prompt = input("Enter prompt here: ")

results, selected_indices, selected_embeddings, MLP_output, candidates, candidate_embeddings = run_experiment(prompt,
                        model_architecture='LSTM', num_epochs=num_epochs, batch_size=batch_size,
                         learning_rate=learning_rate, num_genes=num_genes, loss_function=lukasiewicz_implication_2,
                        window_length=window_length, database_path=database_path, pretrained_model_path=pretrained_model_path)

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
