6 grouped experiments

2 runs; N_train=1000, N_test=2000, N_lip=1000; Euclidean metric. Formula: fixed L=1; constrained NN bound=1. 3 hidden layers, Adam lr=0.001, 1001 full-batch updates.

Percentages are 100 * model mean MSE / reference mean MSE. Above 100% versus the formula favors the formula. Near ties are inclusive [85%,115%]. Expected groups are fixed from the earlier exploratory runs; they do not force outcomes. The constant predictor is the training-label mean. A single run has no sample standard deviation (shown as N/A).


Raw test MSE: mean +/- sample std; target storage ratio 1

(Expected) Clear wins
| Target | d | Formula | ReLU MLP | Spectral | Orthogonal | Constant |
| --- | --- | --- | --- | --- | --- | --- |
| 64 signed bumps | 2 | 2.235e-04 +/- 1.771e-05 | 1.048e-03 +/- 3.225e-04 | 1.718e-03 +/- 7.050e-05 | 1.643e-03 +/- 1.663e-05 | 2.062e-03 +/- 1.537e-05 |
| Folding: 4 layers | 2 | 9.119e-05 +/- 3.013e-05 | 3.415e-04 +/- 4.991e-05 | 9.302e-03 +/- 7.908e-04 | 6.562e-03 +/- 1.174e-03 | 6.231e-02 +/- 1.164e-02 |
(Expected) Clear near ties [85%-115%]
| Target | d | Formula | ReLU MLP | Spectral | Orthogonal | Constant |
| --- | --- | --- | --- | --- | --- | --- |
| Folding: 1 layer | 100 | 1.315e-01 +/- 5.526e-03 | 1.404e-01 +/- 1.070e-02 | 1.234e-01 +/- 5.545e-03 | 1.261e-01 +/- 4.571e-03 | 1.200e-01 +/- 3.708e-03 |
| Folding: 4 layers | 100 | 5.325e-02 +/- 3.113e-03 | 5.021e-02 +/- 2.624e-03 | 4.918e-02 +/- 2.562e-03 | 4.781e-02 +/- 2.813e-03 | 4.221e-02 +/- 2.679e-03 |
(Expected) Clear losses
| Target | d | Formula | ReLU MLP | Spectral | Orthogonal | Constant |
| --- | --- | --- | --- | --- | --- | --- |
| Affine, \|\|w\|\|_1=1 | 100 | 5.672e-03 +/- 1.372e-04 | 2.863e-04 +/- 1.548e-05 | 1.651e-04 +/- 2.153e-05 | 2.817e-03 +/- 7.342e-05 | 5.260e-03 +/- 1.772e-05 |
| Tanh ridge | 100 | 1.475e-01 +/- 2.278e-02 | 1.148e-02 +/- 2.424e-03 | 1.190e-02 +/- 6.147e-04 | 6.165e-02 +/- 1.605e-02 | 1.725e-01 +/- 4.212e-02 |

MSE as a percentage of the formula; target storage ratio 1

(Expected) Clear wins
| Target | d | Formula | ReLU MLP | Spectral | Orthogonal | Constant |
| --- | --- | --- | --- | --- | --- | --- |
| 64 signed bumps | 2 | 100.0% | 468.7% | 768.4% | 735.1% | 922.6% |
| Folding: 4 layers | 2 | 100.0% | 374.5% | 10200.5% | 7195.6% | 68335.9% |
(Expected) Clear near ties [85%-115%]
| Target | d | Formula | ReLU MLP | Spectral | Orthogonal | Constant |
| --- | --- | --- | --- | --- | --- | --- |
| Folding: 1 layer | 100 | 100.0% | 106.7% | 93.8% | 95.8% | 91.3% |
| Folding: 4 layers | 100 | 100.0% | 94.3% | 92.4% | 89.8% | 79.3% |
(Expected) Clear losses
| Target | d | Formula | ReLU MLP | Spectral | Orthogonal | Constant |
| --- | --- | --- | --- | --- | --- | --- |
| Affine, \|\|w\|\|_1=1 | 100 | 100.0% | 5.0% | 2.9% | 49.7% | 92.7% |
| Tanh ridge | 100 | 100.0% | 7.8% | 8.1% | 41.8% | 117.0% |

MSE as a percentage of the constant predictor; target storage ratio 1

(Expected) Clear wins
| Target | d | Formula | ReLU MLP | Spectral | Orthogonal | Constant |
| --- | --- | --- | --- | --- | --- | --- |
| 64 signed bumps | 2 | 10.8% | 50.8% | 83.3% | 79.7% | 100.0% |
| Folding: 4 layers | 2 | 0.1% | 0.5% | 14.9% | 10.5% | 100.0% |
(Expected) Clear near ties [85%-115%]
| Target | d | Formula | ReLU MLP | Spectral | Orthogonal | Constant |
| --- | --- | --- | --- | --- | --- | --- |
| Folding: 1 layer | 100 | 109.6% | 116.9% | 102.8% | 105.0% | 100.0% |
| Folding: 4 layers | 100 | 126.1% | 118.9% | 116.5% | 113.3% | 100.0% |
(Expected) Clear losses
| Target | d | Formula | ReLU MLP | Spectral | Orthogonal | Constant |
| --- | --- | --- | --- | --- | --- | --- |
| Affine, \|\|w\|\|_1=1 | 100 | 107.8% | 5.4% | 3.1% | 53.6% | 100.0% |
| Tanh ridge | 100 | 85.5% | 6.7% | 6.9% | 35.7% | 100.0% |

Empirical Lipschitz estimate: mean +/- sample std; target storage ratio 1

(Expected) Clear wins
| Target | d | Formula | ReLU MLP | Spectral | Orthogonal | Constant |
| --- | --- | --- | --- | --- | --- | --- |
| 64 signed bumps | 2 | 1.0 +/- 0.0 | 1.1 +/- 0.0 | 0.2 +/- 0.0 | 0.2 +/- 0.1 | 0.0 +/- 0.0 |
| Folding: 4 layers | 2 | 1.0 +/- 0.0 | 1.2 +/- 0.1 | 0.6 +/- 0.1 | 0.7 +/- 0.0 | 0.0 +/- 0.0 |
(Expected) Clear near ties [85%-115%]
| Target | d | Formula | ReLU MLP | Spectral | Orthogonal | Constant |
| --- | --- | --- | --- | --- | --- | --- |
| Folding: 1 layer | 100 | 0.1 +/- 0.0 | 0.2 +/- 0.0 | 0.1 +/- 0.0 | 0.1 +/- 0.0 | 0.0 +/- 0.0 |
| Folding: 4 layers | 100 | 0.1 +/- 0.0 | 0.1 +/- 0.0 | 0.1 +/- 0.0 | 0.1 +/- 0.0 | 0.0 +/- 0.0 |
(Expected) Clear losses
| Target | d | Formula | ReLU MLP | Spectral | Orthogonal | Constant |
| --- | --- | --- | --- | --- | --- | --- |
| Affine, \|\|w\|\|_1=1 | 100 | 0.0 +/- 0.0 | 0.1 +/- 0.0 | 0.0 +/- 0.0 | 0.0 +/- 0.0 | 0.0 +/- 0.0 |
| Tanh ridge | 100 | 0.1 +/- 0.0 | 0.3 +/- 0.0 | 0.2 +/- 0.0 | 0.2 +/- 0.0 | 0.0 +/- 0.0 |

Stored scalars
| d | Method | Width | Scalars | Formula scalars | P / P_formula |
| --- | --- | --- | --- | --- | --- |
| 2 | Formula | N/A | 3001 | 3001 | 1.000000 |
| 2 | ReLU MLP | 37 | 2961 | 3001 | 0.986671 |
| 2 | Spectral MLP | 37 | 2961 | 3001 | 0.986671 |
| 2 | Orthogonal MLP | 37 | 2961 | 3001 | 0.986671 |
| 2 | Training-mean constant | N/A | 1 | 3001 | 0.000333 |
| 100 | Formula | N/A | 101001 | 101001 | 1.000000 |
| 100 | ReLU MLP | 200 | 100801 | 101001 | 0.998020 |
| 100 | Spectral MLP | 200 | 100801 | 101001 | 0.998020 |
| 100 | Orthogonal MLP | 200 | 100801 | 101001 | 0.998020 |
| 100 | Training-mean constant | N/A | 1 | 101001 | 0.000010 |

The affine target has ||w||_1=1; its Euclidean Lipschitz constant is ||w||_2, recorded per seed in records/*_diagnostics.json. Formula and NN bounds are upper bounds, not assertions that the actual predictor constant equals the bound. Scalar storage excludes optimizer state and temporary normalization buffers. Orthogonal constraints reduce independent degrees of freedom despite matching dense scalar storage. The tanh ridge uses ||w||_2=1 and b=-w^T x0 for an interior point x0, so its Euclidean Lipschitz constant is exactly one. It is approximately affine locally where |w^T x+b| is small; the uniform sample is not restricted to that region.
