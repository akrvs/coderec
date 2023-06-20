import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.utils.data as data

def preprocess_data(word_list):
    word_list = [word for sentence in word_list for word in sentence.replace(",", " ").split(" ") if word]
    word_list.append("UNK")

    word_to_int = dict((w, i) for i, w in enumerate(word_list))
    int_to_word = dict((i, w) for w, i in word_to_int.items())
    n_words = len(word_list)

    print(f"Data: {n_words} words")

    return word_list, word_to_int, int_to_word, n_words

def create_sequences(word_list, word_to_int, window_length):
    input_sequences = []
    output_sequences = []

    for i in range(len(word_list) - window_length):
        input_seq = word_list[i:i+window_length]
        output_seq = word_list[i+window_length]

        x = np.zeros((window_length, len(word_list)))
        for j, word in enumerate(input_seq):
            x[j, word_to_int[word]] = 1

        y = np.zeros(len(word_list))
        y[word_to_int[output_seq]] = 1

        input_sequences.append(x)
        output_sequences.append(y)

    array1 = np.array(input_sequences)
    array2 = np.array(output_sequences)

    X = torch.tensor(array1, dtype=torch.float32)
    y = torch.tensor(array2, dtype=torch.long)

    return X, y


def data_loading(X, y, test_size, batch_size):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)

    train_dataset = data.TensorDataset(X_train, y_train)
    val_dataset = data.TensorDataset(X_val, y_val)

    train_loader = data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    val_loader = data.DataLoader(val_dataset, shuffle=True, batch_size=batch_size)

    return train_loader, val_loader


