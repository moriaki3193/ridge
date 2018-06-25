import numpy as np
cimport numpy as np
from ridge.racer import predictors
from ridge.racer import link_functions


# Calculation of loss, depending on the task of the model.

def fm_cross_entropy(float b, np.ndarray w, np.ndarray V,
                       np.ndarray X, np.ndarray y):
    """Given X and y, calculate an `approx` cross entropy loss.

    Parameters
    ----------
    X : np.ndarray, whose shape is (n_samples, n_features).
    y : np.ndarray of {0, +1}, whose shape is (n_samples, ).
    """
    # Variable Definitions
    cdef float loss = 0.0
    cdef float score, proba
    cdef np.ndarray y_nnz_ind = np.nonzero(y)[0]
    cdef np.ndarray y_zero_ind = np.where(y == 0)[0]

    # Loss Calculation
    for i in y_nnz_ind:
        score = predictors.fm(b, w, V, X[i])
        proba = link_functions.sigmoid(score)
        loss -= np.log(proba)
    for j in y_zero_ind:
        score = predictors.fm(b, w, V, X[i])
        proba = link_functions.sigmoid(score)
        loss -= np.log(1.0 - proba)
    
    return loss
