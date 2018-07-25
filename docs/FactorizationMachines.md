# Factorization Machines
## FMRegressor
Factorization Machines proposed by Rendle in 2010.
The model tries to minimize sum of squared errors between an observed value and predicted value in each instance.
The optimization method is Stochastic Gradient Descent.

### Usage
- You can use 3 types of inputs
    - `np.ndarray`
    - `np.matrix`
    - `sparse.csr_matrix`
        - `sparse.csr_matrix` is useful when your dataset is highly sparse.
- You have to use `np.ndarray` as an target array.

```Python
import numpy as np
from scipy import sparse
from ridge.models import FMRegressor

# Dataset
X = np.array([
       #  Users  |     Movies     |    Movie Ratings   | Time | Last Movies Rated
       # A  B  C | TI  NH  SW  ST | TI   NH   SW   ST  |      | TI  NH  SW  ST
        [1, 0, 0,  1,  0,  0,  0,   0.3, 0.3, 0.3, 0,     13,   0,  0,  0,  0 ],
        [1, 0, 0,  0,  1,  0,  0,   0.3, 0.3, 0.3, 0,     14,   1,  0,  0,  0 ],
        [1, 0, 0,  0,  0,  1,  0,   0.3, 0.3, 0.3, 0,     16,   0,  1,  0,  0 ],
        [0, 1, 0,  0,  0,  1,  0,   0,   0,   0.5, 0.5,   5,    0,  0,  0,  0 ],
        [0, 1, 0,  0,  0,  0,  1,   0,   0,   0.5, 0.5,   8,    0,  0,  1,  0 ],
        [0, 0, 1,  1,  0,  0,  0,   0.5, 0,   0.5, 0,     9,    0,  0,  0,  0 ],
        [0, 0, 1,  0,  0,  1,  0,   0.5, 0,   0.5, 0,     12,   1,  0,  0,  0 ],
    ])
y = np.array([5, 3, 1, 4, 5, 1, 5])


# Case 1. Type of X_train is np.ndarray
X_train = X[0:5, :]
y_train = y[0:5]
X_test = X[5:, :]
y_test = y[5:]

# Case 2. Type of X_train is np.matrix
X = np.asmatrix(X)
X_train = X[0:5, :]
y_train = y[0:5]
X_test = X[5:, :]
y_test = y[5:]

# Case 3. Type of X_train is sparse.csr_matrix
X = sparse.csr_matrix(X)
X_train = X[0:5, :]
y_train = y[0:5]
X_test = X[5:, :]
y_test = y[5:]

# You don't have to specify any parameters
# when instanciating FMRegressor.
# All parameters can be specified when fitting.
#     - k : The hyper-parameter of Factorization Machines.
#     - l2 : The scaler weight of L2 regularization.
#     - eta : a.k.a Learning Rate.
#     - n_iter : The number of max iteration.
model = FMRegressor().fit(X_train, y_train, k=4, n_iter=1000)
y_pred: np.ndarray = model.predict(X_test)

# This dataset is very small,
# so the performance is not so good as I expected...
print(f'type of X_train: {type(X_train)}')
print(f' Obs[0]: {y_test[0]}')  # 1
print(f'Pred[0]: {y_pred[0]}')  # 4.01237583
print(f' Obs[1]: {y_test[1]}')  # 5
print(f'Pred[1]: {y_pred[1]}')  # 2.69342279
```
