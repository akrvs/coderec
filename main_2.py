import openai
import neural
from utils.chatgpt_predicting import generate_candidates
from utils.similarities import *
from cluster_analysis import genetic_algorithm
from neural.mlp_training_2 import train_mlp_model

# Global Variables
num_epochs = 30
batch_size = 128
learning_rate = 0.01
k = 5
num_genes = 10
num_permutations = 10

# API key for ChatGPT
openai.api_key = ''

# Generate sentences and their embeddings
initial_prompt = input("Enter prompt here: ")
split_candidate_sentences, embeddings = generate_candidates(initial_prompt)

# Create the MLP model and train it
mlp_model = neural.MlpModel(len(embeddings[0]), 512, len(embeddings[0]))
all_similarity_results = train_mlp_model(mlp_model, embeddings, split_candidate_sentences, learning_rate, num_epochs,
                                 loss_function=lukasiewicz_implication_2)

# Run the genetic_algorithm with the sentences and their corresponding embeddings
embeddings_tensor = torch.tensor(embeddings)
output = mlp_model(embeddings_tensor)
sentence_similarity_results = cosine_similarity(output, output)
results = genetic_algorithm(split_candidate_sentences, sentence_similarity_results, k, num_genes, num_permutations, num_epochs)
print(results)



