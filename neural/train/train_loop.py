import numpy as np
from utils.similarities import cosine_similarity
from neural.train.validation_loop import calculate_val_loss
from neural.embeddings.sampling import sampling
from data.preproccessing import make_batch
import random
import torch

def train_model(model, model_type, optimizer, n_epochs, word_to_int, int_to_word, n_words, sentences=None, train_loader=None,
                val_loader=None, loss_fn=None, word_list=None):
    """
        Trains a neural model using the specified parameters.

        Args:
            model (torch.nn.Module): The neural model to be trained.
            model_type (str): The type of the model, either "lstm" or "mlp".
            optimizer (torch.optim.Optimizer): The optimizer used for training.
            n_epochs (int): The number of training epochs.
            word_to_int (dict): A dictionary mapping words to their corresponding indices.
            int_to_word (dict): A dictionary mapping indices to their corresponding words.
            n_words (int): The total number of unique words.
            sentences (list): List of sentences for training (for LSTM model).
            train_loader (torch.utils.data.DataLoader): DataLoader for training (for MLP model).
            val_loader (torch.utils.data.DataLoader): DataLoader for validation (optional).
            loss_fn: Loss function used for training.
            word_list (list): List of unique words.

        Returns:
            tuple: A tuple containing:
                - best_model (torch.nn.Module): The best-trained model.
                - additional outputs based on the model type (e.g., train_losses, val_losses, similarity_results).
    """
    best_model = None
    best_loss = np.inf
    train_losses = []
    val_losses = []
    similarity_results = []
    accuracy_list = []

    for epoch in range(n_epochs):
        total_loss = 0
        correct = 0
        total = 0
        if model_type == "lstm":
            for _ in range(len(sentences)//128 + 1):
                input_batch, target_batch = make_batch(random.sample(sentences, 128), word_to_int,  n_words)
                optimizer.zero_grad()
                output = model(input_batch)
                loss = loss_fn(output, target_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                _, predicted = torch.max(output.data, 1)
                total += target_batch.size(0)
                correct += (predicted == target_batch).sum().item()

            avg_loss = total_loss / (len(sentences) // 128 + 1)
            accuracy = correct / total
            accuracy_list.append(accuracy)
            train_losses.append(avg_loss)
            print(f"\rEpoch: {epoch + 1}/{n_epochs} -  Train Loss: {loss:.3f} - Accuracy: {accuracy:.03f}", end="")

        elif model_type == "mlp":
                y_pred = model(X_batch)
                similarity = cosine_similarity(y_pred, y_batch)
                similarity_results = similarity
                similarity = similarity.mean()
                loss = similarity
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

    best_model = model.state_dict()

    if model_type == "mlp":
        return best_model, train_losses, val_losses, similarity_results
    else:
        return best_model, word_to_int, int_to_word, train_losses, accuracy_list

