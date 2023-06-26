import random

def sampling(X_batch, word_list, max_sample_size):
    """
    Performs sampling on the input batch and returns the sampled embeddings and corresponding words.

    Args:
        X_batch: Input batch tensor.
        word_list: List of words corresponding to the input batch.
        max_sample_size: Maximum size of the sampled embeddings.

    Returns:
        The words and the corresponding sampled embeddings.
    """
    sample_size = min(X_batch.size(0), max_sample_size)
    sample_indices = random.sample(range(X_batch.size(0)), sample_size)
    sampled_embeddings = X_batch[sample_indices]

    words = [word_list[sample_indices[i]] for i in range(len(sample_indices))]

    return words, sampled_embeddings