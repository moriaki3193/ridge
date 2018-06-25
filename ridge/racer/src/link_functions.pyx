import numpy as np
cimport numpy as np


# A series of link functions.

def sigmoid(float z):
    """Sigmoid function.
    """
    return 1.0 / (1.0 + np.exp(-z))
