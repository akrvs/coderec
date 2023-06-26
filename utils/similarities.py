import torch

def cosine_similarity(X, y):
    """
    Computes the cosine similarity between two tensors.

    Args:
        X: Torch tensor representing the first set of vectors.
        y: Torch tensor representing the second set of vectors.

    Returns:
        Torch tensor representing the cosine similarity matrix.
    """
    X = X.float()
    y = y.float()
    X = torch.nn.functional.normalize(X, p=2, dim=1)
    y = torch.nn.functional.normalize(y, p=2, dim=1)

    return torch.matmul(X, y.t())