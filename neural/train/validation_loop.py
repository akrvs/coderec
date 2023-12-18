import torch
from utils.similarities import cosine_similarity

def calculate_val_loss(model, val_loader, loss_fn=None):
    """
    Calculates the validation loss of a model using the provided validation data.

    Args:
        model: The model for which to calculate the validation loss.
        val_loader: The data loader for the validation data.
        loss_fn: The loss function used for calculating the loss. Default is None.

    Returns:
        The validation loss.
    """
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for X_val_batch, y_val_batch in val_loader:
            X_val_batch = X_val_batch.float()
            y_val_batch = y_val_batch.float()
            y_val_pred = model(X_val_batch)

            if loss_fn is None:
                val_loss += cosine_similarity(y_val_pred, y_val_batch)

            else:
                val_loss += loss_fn(y_val_pred, y_val_batch).item()

            # sample_size = min(X_val_batch.size(0), max_sample_size)
            # sample_indices = random.sample(range(X_val_batch.size(0)), sample_size)
            # sampled_embeddings = X_val_batch[sample_indices]
            # y_val_pred = model(sampled_embeddings)

    return val_loss
