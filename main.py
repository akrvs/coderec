from data.database import read_database
import utils
from neural.mlp_training import train_mlp_model
from utils.load_or_train import load_or_train_lstm_model
import neural
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# TO DOs:
# A) ACCEPT COMMAND-LINE ARGUMENTS USING THE 'argparse' MODULE FOR BOTH THE pretrained_model_path & THE database_path.
# B) GENERAL BETTER ORGANIZATION

n_epochs = 50
batch_size = 128
window_length = 1

# Loading models
pretrained_model_path = "/Users/akrvs/PycharmProjects/Project/lstm_model.pth"
database_path = '/Users/akrvs/PycharmProjects/Project/Isaac_Asimov.txt'
word_list = read_database(database_path)
word_list, word_to_int, int_to_word, n_words = utils.preprocess_data(word_list)
best_lstm_model = load_or_train_lstm_model(neural.LstmModel, database_path, window_length, batch_size, n_epochs,
                                           pretrained_model_path, n_words)
# Loading the candidates
print()
candidates = utils.generate_candidates(initial_prompt=input("Enter a word from the database: "),
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

# Find the best k using Silhouette Score
    '''k_values = range(2, 11)  # Range of k values to consider
silhouette_scores = []
for k in k_values:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(candidate_embeddings)
    cluster_labels = kmeans.labels_
    score = silhouette_score(candidate_embeddings, cluster_labels)
    silhouette_scores.append(score)

# Find the best k value
best_index = silhouette_scores.index(max(silhouette_scores))
best_k = k_values[best_index]

# Print the best k value and Silhouette Score
print(f"Best k: {best_k}")
print(f"Silhouette Score: {silhouette_scores[best_index]}")
'''
# Running the code above, we get k = 9. So for k = 9:
k = 9
kmeans = KMeans(n_clusters=k)
kmeans.fit(candidate_embeddings)
cluster_labels = kmeans.labels_

# Collect candidates for each cluster
cluster_candidates = [[] for _ in range(k)]
for i, candidate in enumerate(candidates):
    cluster_candidates[cluster_labels[i]].append(candidate)

# Print candidates in each cluster
for cluster_id, candidates_in_cluster in enumerate(cluster_candidates):
    print(f"Cluster {cluster_id}:")
    for candidate in candidates_in_cluster:
        print(candidate)
    print()

'''# Extract one representative sentence for each cluster. I choose the sentence that is closest to the centroid of the cluster.
representative_sentences = []
for cluster_id, candidates_in_cluster in enumerate(cluster_candidates):
    centroid = kmeans.cluster_centers_[cluster_id]
    closest_candidate = min(candidates_in_cluster, key=lambda c: utils.cosine_similarity([centroid],
                                                                                         [candidate_embeddings[c]]))
    representative_sentences.append(closest_candidate)

# Print representative sentence for each cluster
for cluster_id, representative_sentence in enumerate(representative_sentences):
    print(f"Cluster {cluster_id}:")
    print(representative_sentence)
    print()'''
