# Change Log

All notable changes to the "snippets-scikit-learn" extension will be documented in this file.

## [1.0.6] 2025-12-05

Added Quantile regression (`sk-regress-linear-quantile`) snippet.

## [1.0.5] 2025-11-23

Fixed default value of `min_dist` parameter in `sk-embed-umap`.

## [1.0.4] 2025-09-18

Added Gaussian process regression (`sk-regress-gaussian`) snippets as follows:

- `sk-regress-gaussian-process`

- `sk-regress-gaussian-process-kernel`

- `sk-regress-gaussian-transform-target`

## [1.0.3] 2025-09-06

Ignore `.github` folder for packaging.

Added UMAP embedding (`sk-embed-umap`) snippet. This algorithm is not a native part of `scikit-learn` and requires the `umap-learn` package.

## [1.0.2] 2025-09-04

Added nearest neighbour regression (`sk-regress-neighbours`) snippets as follows:

- `sk-regress-neighbors-k`  

- `sk-regress-neighbors-radius`  

Added nearest neighbour classification (`sk-classify-neighbors`) snippets as follows:

- `sk-classify-neighbors-k`  

- `sk-classify-neighbors-radius`  

- `sk-classify-neighbors-centroid`

Added Neighborhood Components Analysis embedding (`sk-embed-nca`) snippet.


## [1.0.1] 2025-08-27

Fixed typo on `weights` in `sk-regress-ensemble-voting`.

Changed default value of `alphas` argument from `None` to `100` in `sk-regress-linear-elasticnetcv` to silence deprecation warning.

## [1.0.0] 2025-08-26.

Added voting estimators as follows:

- `sk-regress-ensemble-voting`

 - `sk-classify-ensemble-voting`

## [0.0.9] 2025-07-27. 

Corrected typographic error in README.md snippet description.

Added `sk-classify-linear` and `sk-classify-ensemble` branches.

Rearranged `sk-io` snippets into `pickle` and `skops` branches as follows:

- `sk-io-pickle-read`

- `sk-io-pickle-write`

- `sk-io-skops-read`

- `sk-io-skops-write`

## [0.0.7] 2025-06-30

Added the following features for clustering models:

- `sk-cluster-kmeans`  

- `sk-cluster-kmeans-minibatch`

- `sk-cluster-meanshift`

- `sk-cluster-dbscan`  

- `sk-cluster-hdbscan`  

- `sk-cluster-predict`

## [0.0.5] 2025-06-24

Added the following features for density estimation models:  

- `sk-density-kernel`  

- `sk-density-gaussian-mixture`  

- `sk-density-sample-kernel`  

- `sk-density-sample-gaussian-mixture`  

- `sk-density-score-samples`  

- `sk-density-score`

## [0.0.3] 2025-06-05

- Initial pre-release