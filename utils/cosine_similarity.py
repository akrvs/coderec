import torch

def cosine_similarity(X, y):
    X = X.float()
    y = y.float()
    X = torch.nn.functional.normalize(X, p=2, dim=1)
    y = torch.nn.functional.normalize(y, p=2, dim=1)

    return torch.matmul(X, y.t())