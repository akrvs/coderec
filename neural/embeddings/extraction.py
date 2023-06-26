import torch
import numpy as np

def get_embeddings(embedding_model, train_loader):
    """
    Obtains embeddings for the input data using an embedding model.

    Args:
        embedding_model: The embedding model to use for generating embeddings.
        train_loader: The data loader containing the input data.

    Returns:
        Tensor containing the embeddings for the input data.
    """
    embedding_model.eval()
    embeddings = []
    with torch.no_grad():
        for X_batch, _ in train_loader:
            output, _ = embedding_model(X_batch)
            embeddings.extend(output[:, -1, :].tolist())

    embeddings = np.array(embeddings)
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
    return embeddings_tensor

def get_candidate_embeddings(embedding_model, candidates, window_length, n_words, word_to_int):
    """
        Obtains embeddings for the candidate inputs using an embedding model.

        Args:
            embedding_model: The embedding model to use for generating embeddings.
            candidates: List of candidate inputs.
            window_length: Length of the sliding window used for generating patterns.
            n_words: Number of words in the vocabulary.
            word_to_int: Mapping of words to integer indices.

        Returns:
            Tensor containing the embeddings for the candidate inputs.
        """
    embedding_model.eval()
    candidate_embeddings = []

    with torch.no_grad():
        for candidate in candidates:
            words = candidate[-window_length:]
            if len(words) >= window_length:
                pattern = np.zeros((window_length, n_words))
                for i, w in enumerate(words):
                    pattern[i, word_to_int[w]] = 1
                array1 = np.array([pattern])
                x = torch.tensor(array1, dtype=torch.float32)

                output, _ = embedding_model(x)
                candidate_embeddings.extend(output[:, -1, :].tolist())

    candidate_embeddings = np.array(candidate_embeddings)
    candidate_embeddings = torch.tensor(candidate_embeddings, dtype=torch.float32)

    return candidate_embeddings

