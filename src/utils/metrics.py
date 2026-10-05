import torch
import torch.nn.functional as F

def euclidean_distance(x: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """Standard Mean-Vector routing metric (L2 distance)."""
    return torch.cdist(x, centroids, p=2.0)

def cosine_similarity(x: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """Measures directional alignment, ignoring structural geometry."""
    x_norm = F.normalize(x, p=2, dim=1)
    c_norm = F.normalize(centroids, p=2, dim=1)
    return torch.mm(x_norm, c_norm.t())
