import numpy as np
from utils.similarities import cosine_similarity
from neural.train.validation_loop import calculate_val_loss
# from neural.embeddings.sampling import sampling


def train_model(model, model_type, train_loader, val_loader, optimizer, n_epochs, loss_fn=None, word_list=None):
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
    # max_sample_size = 100
    train_losses = []
    val_losses = []
    similarity_results = []

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            if model_type == "lstm":
                y_batch = y_batch.float()
                X_batch = X_batch.float()
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            elif model_type == "mlp":
                # with open("similarity_results.txt", "a") as result_file:
                    # words, sampled_embeddings = sampling(X_batch, word_list, max_sample_size)
                    # y_pred = model(sampled_embeddings)
                    y_pred = model(X_batch)
                    similarity = cosine_similarity(y_pred, y_batch)

                    similarity_results = similarity

                    # result_file.write("Cosine Similarity:\n")
                    # result_file.write(str(similarity) + "\n")

                    # print("\t")
                    # print("Words being compared: ")
                    # print(words)

                    similarity = similarity.mean()

                    loss = similarity
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        if model_type == "lstm":
            val_loss = calculate_val_loss(model, val_loader, loss_fn)
            avg_val_loss = val_loss / len(val_loader)
        elif model_type == "mlp":
            val_loss = calculate_val_loss(model, val_loader, loss_fn=None)
            avg_val_loss = val_loss / len(val_loader)
            avg_val_loss = avg_val_loss.mean()

        train_losses.append(avg_loss)
        val_losses.append(avg_val_loss)

        print(f"\rEpoch {epoch+1}/{n_epochs} - Train Loss: {avg_loss:.3f} - Val Loss: {avg_val_loss:.3f}", end="")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model = model.state_dict()

    if model_type == "mlp":
        return best_model, train_losses, val_losses, similarity_results
    else:
        return best_model, train_losses, val_losses

