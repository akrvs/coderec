import torch
from sklearn.metrics.pairwise import cosine_similarity

def calculate_average_cosine_similarity(embeddings):
    embeddings_tensor = torch.tensor(embeddings)
    similarities = cosine_similarity(embeddings_tensor, embeddings_tensor)
    num_embeddings = len(embeddings)
    total_similarity = 0
    pairs_count = 0
    for i in range(num_embeddings):
        for j in range(i + 1, num_embeddings):
            total_similarity += similarities[i][j]
            pairs_count += 1
    if pairs_count == 0:
        return 0
    average_similarity = total_similarity / pairs_count
    return average_similarity