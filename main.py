from data.database import read_database
import utils
from neural.mlp_training import train_mlp_model
from utils.load_or_train import load_or_train_lstm_model
import neural
import random
from sklearn.cluster import KMeans

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

'''k = 4
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
'''
n = len(candidates)

def create_gene(k, n): # Eixe thema h .choices kai ebaze ligotera 1 genikotera.
    gene = [0 for _ in range(n)]
    selected_positions = random.sample(range(n), k=k)
    for pos in selected_positions:
        gene[pos] = 1

    return gene

def fitness(gene):
    positions = [i for i, exists in enumerate(gene) if exists == 1]
    ret = 0
    for i in positions:
        for j in positions:
            ret += similarity_results[i][j]
    return ret

def permute(gene):
    while True:
        positions = [i for i, exists in enumerate(gene) if exists == 1]
        pos = positions[int(random.random() * len(positions))]
        new_gene = [value for value in gene]
        new_gene[pos] = 0
        new_gene[int(random.random() * len(new_gene))] = 1
        print(new_gene)
        if sum(new_gene) >= 5:
            return new_gene

pool = [create_gene(5, n) for _ in range(10)]
print(pool[0])
print(permute(pool[0]))

for epoch in range(30):
    new_pool = []
    for gene in pool:
        for _ in range(10):
            new_pool.append(permute(gene))
        new_pool.append(gene)
    evals = {i: -fitness(gene) for i, gene in enumerate(pool)}
    pool_ids = sorted(list(evals.keys()), key=lambda i: evals[i])[:len(pool)]
    pool = [new_pool[i] for i in pool_ids]
    print("Best Fitness: ",  - evals[pool_ids[0]])
    print(evals)
    print(pool_ids)

results = [candidates[i] for i, exists in enumerate(pool[0]) if exists == 1]
print(results)

