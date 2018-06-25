import numpy as np
cimport numpy as np
from cpython cimport array
import array


# Prediction function of models.

def fm(float b, np.ndarray w, np.ndarray V, np.ndarray x):
    """Prediction of FM for Regression Task.

    Parameters
    ----------
    b : float, the bias term of FM model.
    w : np.ndarray, its shape is (n_features, ).
        Point-wise weight term of FM model.
    V : np.ndarray, its shape is (n_features, k).
        A vector of latent vectors of pair-wise interaction.
    x : np.ndarray, its shape is (n_features, ).
    """
    # Variable Definition
    cdef int k = V.shape[1]
    cdef np.ndarray nnz_ind = np.nonzero(x)[0]
    cdef float pointwise_score = np.dot(w[nnz_ind], x[nnz_ind])
    cdef float pairwise_score
    cdef array.array pairwise_scores = array.array('f', [])
    cdef float score
    # Within for loops
    cdef int i, f
    cdef float x_i
    cdef float V_if
    cdef float agg_sum_squared
    cdef float agg_squared_sum
    cdef array.array sum_squared, squared_sum

    # Calculation of Pair-wise interactions.
    for f in np.arange(start=0, stop=k):
        sum_squared = array.array('f', [])
        squared_sum = array.array('f', [])
        for i in nnz_ind:
            x_i = x[i]
            V_if = V[i,f]
            sum_squared.append(V_if * x_i)
            squared_sum.append(np.square(x_i) * np.square(V_if))
        agg_sum_squared = np.square(np.sum(sum_squared))
        agg_squared_sum = np.sum(squared_sum)
        pairwise_scores.append(agg_sum_squared - agg_squared_sum)
    pairwise_score = np.sum(pairwise_scores)

    # Score predicted.
    score = b + pointwise_score + 0.5 * pairwise_score

    return score
