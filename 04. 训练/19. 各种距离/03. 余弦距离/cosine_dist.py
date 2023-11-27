

def compute_cosine_distance(features, others):
    """Computes cosine distance.
    Args:
        features (torch.Tensor): 2-D feature matrix.
        others (torch.Tensor): 2-D feature matrix.
    Returns:
        torch.Tensor: distance matrix.
    """
    features = F.normalize(features, p=2, dim=1)
    others = F.normalize(others, p=2, dim=1)
    dist_m = 1 - torch.mm(features, others.t())
    return dist_m.cpu().numpy()