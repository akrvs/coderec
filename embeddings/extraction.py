import torch
import numpy as np

def get_embeddings(embedding_model, train_loader):
    embedding_model.eval()
    embeddings = []
    with torch.no_grad():
        for X_batch, _ in train_loader:
            output, _ = embedding_model(X_batch)
            embeddings.extend(output[:, -1, :].tolist())

    embeddings = np.array(embeddings)
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
    return embeddings_tensor
