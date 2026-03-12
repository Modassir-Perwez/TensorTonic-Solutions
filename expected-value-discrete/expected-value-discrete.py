import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """

    x = np.array(x, dtype=float)
    p = np.array(p, dtype=float)

    # check same length
    if len(x) != len(p):
        raise ValueError("x and p must have same length")

    # check probability sum
    if not np.isclose(np.sum(p), 1):
        raise ValueError("Probabilities must sum to 1")

    return float(np.sum(x * p))