import numpy as np
import torch
import random
import torch.utils.data as data
from sklearn.model_selection import train_test_split

def preprocess_data(word_list):
    """
    Preprocesses the given word list by creating various needed dictionaries.

    Args:
        word_list: The list of words to preprocess.

    Returns:
        The preprocessed word list, word-to-index dictionary,
        index-to-word dictionary, and the total number of words.
    """
    word_list = [word for sentence in word_list for word in sentence.replace(",", " ").split(" ") if word]

    word_to_int = dict((w, i) for i, w in enumerate(set(word_list)))
    int_to_word = dict((i, w) for w, i in word_to_int.items())
    word_to_int["UNK"] = len(word_to_int)
    n_words = len(word_to_int)

    return word_list, word_to_int, int_to_word, n_words

def create_sequences(word_list, word_to_int, window_length):
    """
    Creates input-output sequences from the given word list
    and one-hot encodes them.

    Args:
        word_list: The preprocessed list of words.
        word_to_int: The word-to-index dictionary.
        window_length: The length of the input window.

    Returns:
        Torch tensors containing the input sequences and the output sequences, after encoding.
    """
    input_sequences = []
    output_sequences = []

    for i in range(len(word_list) - window_length):
        input_seq = word_list[i:i+window_length]
        output_seq = word_list[i+window_length]

        x = np.zeros((window_length, len(word_to_int)))
        for j, word in enumerate(input_seq):
            x[j, word_to_int[word]] = 1

        y = np.zeros(len(word_to_int))
        y[word_to_int[output_seq]] = 1

        input_sequences.append(x)
        output_sequences.append(y)

    array1 = np.array(input_sequences)
    array2 = np.array(output_sequences)

    X = torch.tensor(array1, dtype=torch.float32)
    y = torch.tensor(array2, dtype=torch.long)

    return X, y

def create_sequences_sampled(word_list, word_to_int, window_length, sample_size):
    """
    Creates input-output sequences from the given word list, samples a subset,
    and performs one-hot encoding.

    Args:
        word_list: The preprocessed list of words.
        word_to_int: The word-to-index dictionary.
        window_length: The length of the input window.
        sample_size: The size of the sample to be used.

    Returns:
        Torch tensors containing the input sequences and the output sequences, after encoding.
    """
    input_sequences = []
    output_sequences = []

    # Randomly sample a subset of the data
    sample_indices = random.sample(range(len(word_list) - window_length), sample_size)

    for i in sample_indices:
        input_seq = word_list[i:i + window_length]
        output_seq = word_list[i + window_length]

        x = np.zeros((window_length, len(word_to_int)))
        for j, word in enumerate(input_seq):
            x[j, word_to_int[word]] = 1

        y = np.zeros(len(word_to_int))
        y[word_to_int[output_seq]] = 1

        input_sequences.append(x)
        output_sequences.append(y)

    array1 = np.array(input_sequences)
    array2 = np.array(output_sequences)

    X = torch.tensor(array1, dtype=torch.float32)
    y = torch.tensor(array2, dtype=torch.long)

    return X, y



def data_loading(X, y, test_size, batch_size):
    """
    Performs data loading and splitting into train and validation sets.

    Args:
        X: Torch tensor containing the input sequences.
        y: Torch tensor containing the output sequences.
        test_size: The proportion of data to be used for testing.
        batch_size: The batch size for data loading.

    Returns:
        Training and validation data loaders.
    """
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)

    train_dataset = data.TensorDataset(X_train, y_train)
    val_dataset = data.TensorDataset(X_val, y_val)

    train_loader = data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    val_loader = data.DataLoader(val_dataset, shuffle=True, batch_size=batch_size)

    return train_loader, val_loader


