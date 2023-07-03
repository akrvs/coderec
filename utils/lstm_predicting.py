import numpy as np
import torch

def generate_candidates(initial_prompt, window_length, n_words, word_to_int, int_to_word, best_lstm_model):
    """
    Generates candidate sequences based on an initial prompt using an LSTM model.

    Args:
        initial_prompt: The initial prompt for generating candidates.
        window_length: Length of the sliding window used for generating patterns.
        n_words: Number of words in the vocabulary.
        word_to_int: Mapping of words to integer indices.
        int_to_word: Mapping of integer indices to words.
        best_lstm_model: The best trained LSTM model.

    Returns:
        A list of generated candidate sequences.
    """
    initial_prompt = initial_prompt.lower()
    candidates = [[initial_prompt]]

    if window_length > 1:
        for position in range(window_length - 1):
            new_candidates = []
            for candidate in candidates:
                assert isinstance(candidate, list)
                words = [""] * (window_length - len(candidate)) + candidate
                pattern = np.zeros((window_length, n_words))
                for i, w in enumerate(words):
                    pattern[i, word_to_int.get(w, word_to_int["UNK"])] = 1
                array1 = np.array([pattern])
                x = torch.tensor(array1, dtype=torch.float32)

                prediction = best_lstm_model(x)
                top_indices = torch.argsort(prediction, descending=True)[0, :5]
                top_predictions = [int_to_word.get(int(idx), "UNK") for idx in top_indices]

                new_candidates.extend(candidate+[word] for word in top_predictions)
            candidates = new_candidates

    else:
        for position in range(2):
            new_candidates = []
            for candidate in candidates:
                assert isinstance(candidate, list)
                words = candidate[-window_length:]
                if len(words) >= window_length:
                    pattern = np.zeros((window_length, n_words))
                    for i, w in enumerate(words):
                        pattern[i, word_to_int[w]] = 1
                    array1 = np.array([pattern])
                    x = torch.tensor(array1, dtype=torch.float32)

                    prediction = best_lstm_model(x)
                    top_indices = torch.argsort(prediction, descending=True)[0, :5]
                    top_predictions = [int_to_word.get(int(idx), "UNK") for idx in top_indices]

                    new_candidates.extend(candidate + [word] for word in top_predictions)
            candidates = new_candidates

    return candidates







