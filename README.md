# ridge
Python Machine Learning Library specialized in L2R and Recommendation.
Named after my favorite racing game `R4`.

[日本語のREADME](./README.ja.md)はこちら．

## Requirements
- Python 3.6.0 ~
- NumPy
- Scipy

## Implementation
These models are available.

| Class | Module | Description | Document |
|:------|:-------|:------------|:---------|
| FMRegressor | factorization_machines | Factorization Machine for regression tasks | [FMs](./docs/FactorizationMachines.md) |
| FMClassifier | factorization_machines | Factorization Machine for classification tasks | [FMs](./docs/FactorizationMachines.md) |
| MatFac | matrix_factorization | Ordinal Matrix Factorization | [MFs](./docs/MatrixFactorization.md) |
| NNMatFac | matrix_factorization | Non-negative Matrix Factorization | [MFs](./docs/MatrixFactorization.md) |
| ConditionalLogit | logit_models | Conditional Logistic Regression as a discrete choice model | [LMs](./docs/LogitModels.md) |

## Directories
| name | descripiton |
|:----:|:------------|
| ridge | Model implementation |
| ridge.racer | Cython optimization |
| docs | Documents (usage of this package & memo related to implementation) |
| tests | Model test suites |

## Models
### MFs
#### MatFac (Matrix Factorization)
- MovieLens 100k
  - with Cython 17m20s (1000epochs)
  - with Python 28m38s (1000epochs)

### FMs
#### FMRegressor (Factorization Machine for Regression Tasks)

#### FMClassifier (Factorization Machine for Classification Tasks)
