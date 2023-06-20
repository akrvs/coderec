import torch
import random
from utils.cosine_similarity import cosine_similarity

def calculate_val_loss(model, val_loader, loss_fn, max_sample_size=None):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for X_val_batch, y_val_batch in val_loader:
            X_val_batch = X_val_batch.float()
            y_val_batch = y_val_batch.float()

            if max_sample_size is not None:
                sample_size = min(X_val_batch.size(0), max_sample_size)
                sample_indices = random.sample(range(X_val_batch.size(0)), sample_size)
                sampled_embeddings = X_val_batch[sample_indices]
                y_val_pred = model(sampled_embeddings)
                val_loss += cosine_similarity(y_val_pred, y_val_pred)

            else:
                y_val_pred = model(X_val_batch)
                val_loss += loss_fn(y_val_pred, y_val_batch).item()

    return val_loss
