import random

def sampling(X_batch, word_list, max_sample_size):
    sample_size = min(X_batch.size(0), max_sample_size)
    sample_indices = random.sample(range(X_batch.size(0)), sample_size)
    sampled_embeddings = X_batch[sample_indices]

    # Get the words being compared
    words = [word_list[sample_indices[i]] for i in range(len(sample_indices))]

    return words, sampled_embeddings