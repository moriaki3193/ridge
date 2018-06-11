import numpy as np
cimport numpy as np

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
