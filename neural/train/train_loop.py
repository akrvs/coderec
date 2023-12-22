import numpy as np
from utils.similarities import cosine_similarity
from neural.train.validation_loop import calculate_val_loss
from neural.embeddings.sampling import sampling
from data.preproccessing import make_batch
import random
import torch

def train_model(model, model_type, optimizer, n_epochs, word_to_int, int_to_word, n_words, sentences=None, train_loader=None, val_loader=None, loss_fn=None, word_list=None):
    """
    Trains a specified model using the provided data and returns the best model, along with the train and
    validation losses.

    Args:
        model: The model to train.
        model_type: Type of the model ("lstm" or "mlp").
        train_loader: The data loader for the train data.
        val_loader: The data loader for the validation data.
        optimizer: The optimizer used for train the model.
        n_epochs: The total number of epochs to train for.
        loss_fn: The loss function used for calculating the loss. Default is None.
        word_list: The list of words corresponding to the input data. Default is None.

    Returns:
        The best model obtained during train, and lists containing train and validation losses.
        If model_type is "mlp", it also returns the similarity results.
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


        '''avg_loss = total_loss / len(train_loader)

        if val_loader is not None:
            val_loss = calculate_val_loss(model, val_loader, loss_fn)
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

        train_losses.append(avg_loss)
        val_losses.append(avg_val_loss)

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model = model.state_dict()'''

    best_model = model.state_dict()

    if model_type == "mlp":
        return best_model, train_losses, val_losses, similarity_results
    else:
        return best_model, word_to_int, int_to_word, train_losses, accuracy_list

