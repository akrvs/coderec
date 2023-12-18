import openai
from utils import generate_candidates_chatgpt
from utils.average_cosine_similarity import calculate_average_cosine_similarity
from flask_app.run_experiments import run_experiment
from utils.similarities import lukasiewicz_implication_2

# Global Variables
batch_size = 128
learning_rate = 0.01
num_genes = 100

# Paths for the LSTM model
pretrained_model_path = "/Users/akrvs/PycharmProjects/Project/lstm_model.pth"
database_path = '/Users/akrvs/PycharmProjects/Project/Isaac_Asimov.txt'

# API key for ChatGPT
openai.api_key = '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'

# User's prompt
prompt = input("Enter prompt here: ")

split_candidate_sentences, embeddings = generate_candidates_chatgpt(prompt)
print(split_candidate_sentences)

results = run_experiment(prompt=prompt, model_architecture='ChatGPT', num_epochs=100, batch_size=batch_size,
                         learning_rate=learning_rate, num_genes=num_genes, loss_function=lukasiewicz_implication_2,
                         window_length=None, database_path=None, pretrained_model_path=None)
print(results)

'''selected_indices, selected_embeddings, MLP_output = run_experiment(prompt=prompt, model_architecture='ChatGPT', num_epochs=100,
                                                              batch_size=batch_size, learning_rate=learning_rate,
                                                            num_genes=num_genes, loss_function=lukasiewicz_implication_2,
                                                            window_length=None, database_path=None, pretrained_model_path=None)

print("Most Dissimilar Sentences:")
for i, indices in enumerate(selected_indices):
    print(f"{i + 1}. {[split_candidate_sentences[idx] for idx in indices]}")

average_cosine_similarity = calculate_average_cosine_similarity(embeddings)
print("Average cosine similarity for ChatGPT:", average_cosine_similarity)

average_cosine_similarity = calculate_average_cosine_similarity(MLP_output)
print("Average cosine similarity after the MLP:", average_cosine_similarity)

average_cosine_similarity = calculate_average_cosine_similarity(selected_embeddings)
print("Average cosine similarity after the Genetic Algorithm:", average_cosine_similarity)'''

# Provide me with 25 questions that contain the words of the phrase: "web service Flask". Each answer should be separated by the "#" character.





