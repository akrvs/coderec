import numpy as np
import torch
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

def pad_sequence(sentence, max_len):
    """
    Pads a given sentence with '<PAD>' tokens to achieve a specified maximum length.

    Args:
        sentence (str): The input sentence to be padded.
        max_len (int): The desired maximum length of the padded sentence.

    Returns:
        str: The padded sentence.
    """
    words = sentence.split()
    padded_words = words + ['<PAD>'] * (max_len - len(words))

    return ' '.join(padded_words)


def tokenizer(sentences):
    """
    Tokenizes a list of sentences, creating dictionaries to map words to indices and vice versa.

    Args:
        sentences (list): A list of sentences to be tokenized.

    Returns:
        tuple: A tuple containing:
            - word_list (list): A list of unique words in the sentences.
            - word_dict (dict): A dictionary mapping words to their corresponding indices.
            - number_dict (dict): A dictionary mapping indices to their corresponding words.
            - n_class (int): The total number of unique words in the sentences.
    """
    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))
    word_dict = {w: i for i, w in enumerate(word_list)}
    word_dict["UNK"] = len(word_dict)
    number_dict = {i: w for i, w in enumerate(word_list)}
    n_class = len(word_dict)

    return word_list, word_dict, number_dict, n_class

def make_batch(sentences, word_dict, n_class):
    """
    Creates input and target batches for training neural networks.

    Args:
        sentences (list): A list of sentences to create batches from.
        word_dict (dict): A dictionary mapping words to their corresponding indices.
        n_class (int): The total number of unique words in the sentences.

    Returns:
        tuple: A tuple containing:
            - input_batch (torch.Tensor): The input batch in one-hot encoded format.
            - target_batch (torch.Tensor): The target batch in index format.
    """
    input_batch = []
    target_batch = []

    for sentence in sentences:
        words = sentence.split()
        input_idx = [word_dict[word] for word in words[:-1]]
        target = word_dict[words[-1]]
        input_one_hot = np.zeros((len(input_idx), n_class), dtype=np.float32)
        input_one_hot[np.arange(len(input_idx)), input_idx] = 1
        input_batch.append(input_one_hot)
        target_batch.append(target)

    return torch.Tensor(input_batch), torch.LongTensor(target_batch)


def remove_question_marks(sentence):
    """
    Removes question marks from a given sentence.

    Args:
        sentence (str): The input sentence containing question marks.

    Returns:
        str: The sentence with question marks removed.
    """
    return sentence.replace("?", "")
