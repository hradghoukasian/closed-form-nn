Noisy 100D targets; 2 seeds, N_train=1000, N_test=2000. Clean test MSE.
Cell: mean +/- sample std [percentage of formula's mean MSE].
Above 100% favors the formula. Groups identify targets, not measured outcomes.



Sigma=0.1; target MLP storage ratio=1


Hard
| Target | Formula | ReLU MLP | SN MLP | Orthogonal MLP | Constant |
| --- | --- | --- | --- | --- | --- |
| 64 signed Voronoi tents | 0.04846 +/- 0.00002 [100.0%] | 0.05898 +/- 0.00042 [121.7%] | 0.05641 +/- 0.00037 [116.4%] | 0.05263 +/- 0.00042 [108.6%] | 0.04847 +/- 0.00003 [100.0%] |

Tied
| Target | Formula | ReLU MLP | SN MLP | Orthogonal MLP | Constant |
| --- | --- | --- | --- | --- | --- |
| Folding: 1 layer | 0.12013 +/- 0.00382 [100.0%] | 0.14236 +/- 0.01310 [118.5%] | 0.12443 +/- 0.00635 [103.6%] | 0.12578 +/- 0.00509 [104.7%] | 0.11998 +/- 0.00362 [99.9%] |
| Folding: 4 layers | 0.04225 +/- 0.00278 [100.0%] | 0.05436 +/- 0.00261 [128.7%] | 0.05198 +/- 0.00220 [123.0%] | 0.04903 +/- 0.00264 [116.0%] | 0.04219 +/- 0.00269 [99.9%] |


Sigma=0.2; target MLP storage ratio=1


Hard
| Target | Formula | ReLU MLP | SN MLP | Orthogonal MLP | Constant |
| --- | --- | --- | --- | --- | --- |
| 64 signed Voronoi tents | 0.04843 +/- 0.00009 [100.0%] | 0.06945 +/- 0.00265 [143.4%] | 0.06257 +/- 0.00016 [129.2%] | 0.05758 +/- 0.00018 [118.9%] | 0.04844 +/- 0.00009 [100.0%] |

Tied
| Target | Formula | ReLU MLP | SN MLP | Orthogonal MLP | Constant |
| --- | --- | --- | --- | --- | --- |
| Folding: 1 layer | 0.12018 +/- 0.00389 [100.0%] | 0.15409 +/- 0.01555 [128.2%] | 0.12953 +/- 0.00717 [107.8%] | 0.12783 +/- 0.00548 [106.4%] | 0.11996 +/- 0.00358 [99.8%] |
| Folding: 4 layers | 0.04229 +/- 0.00278 [100.0%] | 0.06687 +/- 0.00439 [158.1%] | 0.05894 +/- 0.00230 [139.4%] | 0.05301 +/- 0.00362 [125.3%] | 0.04221 +/- 0.00265 [99.8%] |


Sigma=0.8; target MLP storage ratio=1


Hard
| Target | Formula | ReLU MLP | SN MLP | Orthogonal MLP | Constant |
| --- | --- | --- | --- | --- | --- |
| 64 signed Voronoi tents | 0.04951 +/- 0.00005 [100.0%] | 0.27509 +/- 0.00030 [555.7%] | 0.11342 +/- 0.00214 [229.1%] | 0.11185 +/- 0.00372 [225.9%] | 0.04909 +/- 0.00066 [99.2%] |

Tied
| Target | Formula | ReLU MLP | SN MLP | Orthogonal MLP | Constant |
| --- | --- | --- | --- | --- | --- |
| Folding: 1 layer | 0.12058 +/- 0.00450 [100.0%] | 0.35013 +/- 0.00461 [290.4%] | 0.17403 +/- 0.00742 [144.3%] | 0.17155 +/- 0.01255 [142.3%] | 0.12062 +/- 0.00444 [100.0%] |
| Folding: 4 layers | 0.04352 +/- 0.00187 [100.0%] | 0.29630 +/- 0.00903 [680.8%] | 0.10893 +/- 0.00322 [250.3%] | 0.10749 +/- 0.00026 [247.0%] | 0.04313 +/- 0.00132 [99.1%] |
