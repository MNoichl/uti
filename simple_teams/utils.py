import numpy as np


def random_argmax(b):
    """Returns the position of the maximum element, or a random one of
    multiple maxima, if present:
    basically argmax with tiebreaking.
    Args:
        b (list or array):
    Returns:
        int: position of chosen maximum
    """
    return np.random.choice(np.flatnonzero(b == np.max(b)))