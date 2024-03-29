import numpy as np
import torch
import pickle
import re
import os
from neural.architectures.chatgpt_model import get_completion
from transformers import BartModel, BartTokenizer

def predict(custom_input, word_to_int, int_to_word, best_lstm_model, n_words):
    """
    Predicts the next word based on a custom input using an LSTM model.

    Args:
        custom_input (str): The custom input text.
        word_to_int (dict): A dictionary mapping words to their corresponding indices.
        int_to_word (dict): A dictionary mapping indices to their corresponding words.
        best_lstm_model (torch.nn.Module): The trained LSTM model.
        n_words (int): The total number of unique words.

    Returns:
        str: The predicted next word.
    """
    custom_input_tokens = custom_input.split()
    input_idx = [word_to_int[word] for word in custom_input_tokens]

    input_one_hot = np.zeros((len(input_idx), n_words), dtype=np.float32)
    input_one_hot[np.arange(len(input_idx)), input_idx] = 1
    input_one_hot = torch.Tensor(input_one_hot).unsqueeze(0)

    with torch.no_grad():
        output = best_lstm_model(input_one_hot)

    predicted_idx = torch.argmax(output, dim=1).item()
    predicted_word = int_to_word[predicted_idx]
    return predicted_word

def generate_candidates_lstm(initial_prompt, window_length, n_words, word_to_int, int_to_word, best_lstm_model):
    """
        Generates a list of candidate sentences using an LSTM model.

        Args:
            initial_prompt (str): The initial prompt for generating candidates.
            window_length (int): The window length for LSTM model.
            n_words (int): The total number of unique words.
            word_to_int (dict): A dictionary mapping words to their corresponding indices.
            int_to_word (dict): A dictionary mapping indices to their corresponding words.
            best_lstm_model (torch.nn.Module): The trained LSTM model.

        Returns:
            list: A list of candidate sentences.
    """
    initial_prompt = initial_prompt.lower().split()
    candidates = [initial_prompt]

    for position in range(3):
        new_candidates = []
        for candidate in candidates:
            assert isinstance(candidate, list)
            words = candidate[-min(window_length, len(candidate)):]
            print(words)
            pattern = np.zeros((len(words), n_words))

            for i, w in enumerate(words):
                pattern[i, word_to_int[w]] = 1

            array1 = np.array([pattern])
            x = torch.tensor(array1, dtype=torch.float32)

            prediction = best_lstm_model(x)
            top_indices = torch.argsort(prediction, descending=True)[0, :5]
            top_predictions = [int_to_word.get(int(idx), "UNK") for idx in top_indices]

            user_input = ' '.join(candidate)
            predicted_word = predict(user_input, word_to_int, int_to_word, best_lstm_model, n_words)
            top_predictions.append(predicted_word)

            new_candidates.extend(candidate + [word] for word in top_predictions)
        candidates = new_candidates

    return candidates


def generate_candidates_chatgpt(initial_prompt):
    """
        Generates a list of candidate sentences using the ChatGPT model.

        Args:
            initial_prompt (str): The initial prompt for generating candidates.

        Returns:
            tuple: A tuple containing:
                - split_candidate_sentences (list): List of split candidate sentences.
                - embeddings (list): List of embeddings for candidate sentences.
    """
    initial_prompt = initial_prompt.lower()

    candidates_filename = f"/Users/akrvs/PycharmProjects/Project/candidates.pkl"
    if os.path.exists(candidates_filename):
        with open(candidates_filename, "rb") as f:
            split_candidate_sentences, embeddings = pickle.load(f)
    else:
        response = get_completion(initial_prompt, num_candidates=1)
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        model = BartModel.from_pretrained('facebook/bart-base')
        candidate_responses = [choice['message']['content'] for choice in response['choices']]
        candidate_sentences = [re.split(r'#', response) for response in candidate_responses]
        split_candidate_sentences = [[sentence] for sublist in candidate_sentences for sentence in sublist]

        embeddings = []
        for sentence in split_candidate_sentences:
            tokens = tokenizer(sentence[0], return_tensors="pt",
                               max_length=1024, truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**tokens)
            sentence_embeddings = outputs.last_hidden_state.mean(dim=1)
            flattened_embedding = sentence_embeddings.flatten().tolist()
            embeddings.append(flattened_embedding)

        with open(candidates_filename, "wb") as f:
            pickle.dump((split_candidate_sentences, embeddings), f)

    return split_candidate_sentences, embeddings
