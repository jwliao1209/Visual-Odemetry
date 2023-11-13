import numpy as np


def expand_vector_dim(points):
    """
    Expands a vector from n dimensions to n+1 dimensions.
    """
    return np.hstack([points, np.ones((points.shape[0], 1))])


def reduce_vector_dim(points):
    """
    Reduces a vector from n+1 dimensions to n dimensions.
    """
    EPSILON = 1e-8
    dim = points.shape[-1]
    points = points / (np.expand_dims(points[:, dim-1], axis=1) + EPSILON)
    return points[:, :dim-1]
