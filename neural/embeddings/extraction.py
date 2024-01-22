import torch
import numpy as np

def get_candidate_embeddings(embedding_model, candidates, window_length, n_words, word_to_int):
    """
    Generates embeddings for the candidate inputs using the specified embedding model.

    Args:
        embedding_model: The embedding model used to generate the embeddings.
        candidates: A list of candidate inputs.
        window_length: The length of the sliding window used for generating patterns.
        n_words: The number of words in the vocabulary.
        word_to_int: A mapping of words to integer indices.

    Returns:
        A tensor containing the embeddings for the candidate inputs.
    """
    embedding_model.eval()
    candidate_embeddings = []

    with torch.no_grad():
        for candidate in candidates:
            words = [""] * (window_length - len(candidate)) + candidate[-window_length:]
            pattern = np.zeros((window_length, n_words))
            for i, w in enumerate(words):
                pattern[i, word_to_int.get(w, word_to_int["UNK"])] = 1
            array1 = np.array([pattern])
            x = torch.tensor(array1, dtype=torch.float32)

            _, output = embedding_model(x)
            output = output[1]
            candidate_embeddings.append(output.flatten().tolist())

    candidate_embeddings = np.array(candidate_embeddings)
    candidate_embeddings = torch.tensor(candidate_embeddings, dtype=torch.float32)

    return candidate_embeddings