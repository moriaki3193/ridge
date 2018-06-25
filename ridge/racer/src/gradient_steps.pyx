import numpy as np
cimport numpy as np
from ridge.racer import (
    predictors,
    link_functions
)


# Gradient Steps of 1 epoch.

def mf(np.ndarray ratings, np.ndarray P, np.ndarray Q, float alpha):
    # [START Variable definitions]
    errors = []
    cdef int k = P.shape[0]
    cdef long int n_users = ratings.shape[0]
    cdef long int n_items = ratings.shape[1]
    cdef float residue
    # [END Variable definitions]

    # [START Calculation]
    for i in range(n_users):
        for j in np.nonzero(ratings[i, :])[0]:
            p = P[:, i]
            q = Q[:, j]
            residue = ratings[i, j] - np.dot(p, q)
            for f in range(k):
                P[f, i] += alpha * (2 * residue * Q[f][j])
                Q[f, j] += alpha * (2 * residue * P[f][i])
            errors.append(residue)
    # [END Calculation]
    
    return (P, Q, errors)

def fm_classification(float b, np.ndarray w, np.ndarray V, int k,
                      float l2, float eta, np.ndarray X, np.ndarray y):
    """Gradient Steps for FM Classification Task.

    TODO Too Slow !!!
    """
    # Variable Definition.
    cdef int m, f, i
    cdef float obs, coef, shared_term
    cdef np.ndarray row, sample_indices, nnz_ind
    cdef float b_new = b
    cdef np.ndarray w_new = w
    cdef np.ndarray V_new = V
    # Calculation of gradients.
    sample_indices = np.arange(start=0, stop=len(y))
    np.random.shuffle(sample_indices)
    for m in sample_indices:
        row = X[m]
        obs = y[m]
        # Parameter Updation
        z = predictors.fm(b, w, V, row)
        proba = link_functions.sigmoid(z)
        coef = proba - obs
        b_new -= eta * (coef + l2 * b)
        w_new -= eta * (coef * row + l2 * w)
        for f in np.arange(start=0, stop=k):
            nnz_ind = np.nonzero(row)[0]
            shared_term = np.sum([V[j,f] * row[j] for j in nnz_ind])
            for i in nnz_ind:
                first_term = row[i] * shared_term
                second_term = V[i,f] * np.square(row[i])
                V_new[i,f] -= eta * (coef * (first_term - second_term) + l2 * V_new[i,f])

    return (b_new, w_new, V_new)
