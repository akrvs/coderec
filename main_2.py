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
# Βάζω σαν prompt το: Provide me with 30 different answers, separated by “#”, to the question: To be or not to be
# Έκανα και το hash να λειτουργήσει για να μην τρώμε resources από το API της openai.
initial_prompt = input("Enter prompt here: ")
split_candidate_sentences, embeddings = generate_candidates(initial_prompt)

# Create the MLP model and train it
mlp_model = neural.MlpModel(len(embeddings[0]), 512, len(embeddings[0]))
all_similarity_results = train_mlp_model(mlp_model, embeddings, split_candidate_sentences, learning_rate, num_epochs,
                                 loss_function=lukasiewicz_implication_2)

# Run the genetic_algorithm with the sentences and their corresponding embeddings
# 1ος τρόπος (νομίζω είναι λάθος, δεν χρησιμοποιούμε κάπου το trained MLP, οπότε δεν feedάρουμε το σωστό πράγμα στον γενετικό.)
# Πάντως δεν κρασάρει
embeddings_tensor = torch.tensor(embeddings)
output = mlp_model(embeddings_tensor)
sentence_similarity_results = cosine_similarity(output, output)
results = genetic_algorithm(split_candidate_sentences, sentence_similarity_results, k, num_genes, num_permutations, num_epochs)
print(results)

# 2ος τρόπος (αυτός είναι σωστός, αλλά παίζει θέμα με το τύπο που είναι το all_similarity_results)
'''results = genetic_algorithm(split_candidate_sentences, all_similarity_results, k, num_genes, num_permutations, num_epochs)
print(results)'''

# Αυτό το έκανα για debugging για να δω τι έχει μέσα το all_similarity_results, ήταν λίστα με tensors μαζί με τα gradients,
# οπότε τους έκανα detach για να δω αν μπορώ να τα χρησιμοποιήσω στον γενετικό, αλλά παίρνω το ίδιο error
'''for epoch, similarity_results in enumerate(all_similarity_results):
    print(f"Epoch {epoch + 1}:")
    for triplet_similarities in similarity_results:
        detached_similarities = [similarity.detach().item() for similarity in triplet_similarities]
        print(detached_similarities)'''
