Noisy 100D targets; 30 seeds, N_train=1000, N_test=2000. Clean test MSE.
Cell: mean +/- sample std [percentage of formula's mean MSE].
Above 100% favors the formula. Groups identify targets, not measured outcomes.



Sigma=0.2; target MLP storage ratio=1


Hard
| Target | Formula | ReLU MLP | SN MLP | Orthogonal MLP | Constant |
| --- | --- | --- | --- | --- | --- |
| 64 signed Voronoi tents | 0.04920 +/- 0.00272 [100.0%] | 0.06672 +/- 0.00257 [135.6%] | 0.06085 +/- 0.00268 [123.7%] | 0.05649 +/- 0.00272 [114.8%] | 0.04926 +/- 0.00273 [100.1%] |

Tied
| Target | Formula | ReLU MLP | SN MLP | Orthogonal MLP | Constant |
| --- | --- | --- | --- | --- | --- |
| Folding: 1 layer | 0.12059 +/- 0.00458 [100.0%] | 0.14999 +/- 0.00596 [124.4%] | 0.12936 +/- 0.00506 [107.3%] | 0.12967 +/- 0.00524 [107.5%] | 0.12045 +/- 0.00464 [99.9%] |
| Folding: 4 layers | 0.04109 +/- 0.00225 [100.0%] | 0.06474 +/- 0.00358 [157.5%] | 0.05712 +/- 0.00352 [139.0%] | 0.05260 +/- 0.00247 [128.0%] | 0.04105 +/- 0.00223 [99.9%] |


Sigma=0.8; target MLP storage ratio=1


Hard
| Target | Formula | ReLU MLP | SN MLP | Orthogonal MLP | Constant |
| --- | --- | --- | --- | --- | --- |
| 64 signed Voronoi tents | 0.05000 +/- 0.00287 [100.0%] | 0.27795 +/- 0.01516 [555.8%] | 0.10834 +/- 0.00666 [216.7%] | 0.10874 +/- 0.00537 [217.5%] | 0.04972 +/- 0.00277 [99.4%] |

Tied
| Target | Formula | ReLU MLP | SN MLP | Orthogonal MLP | Constant |
| --- | --- | --- | --- | --- | --- |
| Folding: 1 layer | 0.12115 +/- 0.00495 [100.0%] | 0.37155 +/- 0.02820 [306.7%] | 0.18004 +/- 0.00742 [148.6%] | 0.17767 +/- 0.00698 [146.7%] | 0.12095 +/- 0.00489 [99.8%] |
| Folding: 4 layers | 0.04222 +/- 0.00315 [100.0%] | 0.29354 +/- 0.02512 [695.3%] | 0.10673 +/- 0.00769 [252.8%] | 0.10591 +/- 0.00512 [250.8%] | 0.04158 +/- 0.00254 [98.5%] |
