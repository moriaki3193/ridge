# ridge
Python Machine Learning Library specialized in L2R and Recommendation.
Named after my favorite racing game `R4`.

## Requirements
- Python 3.6.0 ~

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
