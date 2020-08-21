# Cluster-HyperEnsembles
Ensembles and hyperparameter optimization for clustering pipelines.

A clustering pipeline consists of learning and applying a feature transformation,
computing distances between observations, and fitting a clustering model.
Any parameters of the feature transformation, distance computation and
clustering algorithm can be included in the ensemble hyperparameters.

Commonly used feature transformations, distance metrics and clustering algorithms
are supported. Any step of the pipeline can be extended by passing
custom functions to the ensemble.

## Consensus Clustering
Cluster assignments from multiple runs are combined with a consensus clustering model.
The consensus is computed by constructing a similarity matrix of co-occurrences
from cluster assignments, and fitting a hierarchical clustering model on the
similarity matrix. The silhouette score is used to estimate the number of clusters for the final clustering.

## Hyperparameter Search
An alternative to clustering aggregation is searching for a single configuration
that optimizes a clustering evaluation metric. Commonly used intrinsic and extrinsic clustering
evaluation metrics can be used in a randomized or pre-specified configuration search.

## Specifying a HyperEnsemble
A HyperEnsemble can be fully specified with three dictionaries: *options*, *parameters* and *definitions*.

### options
A dictionary mapping parameter keys to parameter values for the clustering algorithms,
feature transformations, distance computations, and the rate of observation subsampling. 
A categorical distribution over values can be specified by
setting a parameter value with a dict containing either the `choices`
or `irange` key. The `weights` key can be used to define the probability
of each item, or omitted for a uniform distribution.

Examples: \
`options = {'n_clusters': {'irange': [2, 5]}}` \
The `n_clusters` value will be chosen among 2, 3 or 4 with equal probability.\
`options = {'n_clusters': {'choices': [2, 3, 4], 'weights': [1, 2, 1]}}`\
The `n_clusters` value will be chosen among 2, 3 or 4 with probability proportional to the values in the `weights` list.

### parameters
A dictionary specifying the dependencies between parameters in the options dict.
For example, `parameters = {'hac': ['n_clusters', 'linkage', 'metric']}`
means that option `hac` requires `n_clusters`, `linkage` and `metric` to be specified.
Providing a parameters dict extends and/or overwrites items in the default parameters dict.\
The default parameters dict is
```
parameters = {'hac': ['linkage', 'n_clusters', 'metric'],
              'hacsingle': ['n_clusters', 'metric'],
              'hacaverage': ['n_clusters', 'metric'],
              'kmeans': ['n_clusters'],
              'mbkmeans': ['n_clusters'],
              'dbscan': ['eps', 'min_samples', 'metric'],
              'rbf': ['gamma'],
              'svd': ['n_components'],
              'autoencoder': ['layer_dims'],
              'convautoencoder': ['kernel_shapes', 'h_dim']}
```

### definitions
A dictionary specifying custom transformation and clustering functions.
Providing a definitions dict extends and/or overwrites items in the default definitions dict.
Default function definitions are provided for all the keys in the default parameters dict.

### Custom function definition
Any custom function passed to the definitions dict should implement the following signature \
`customfunc(params, X)`\
where `params` is a dict mapping parameter keys to values and `X` is the data matrix.
Feature transformations should return an ndarray of shape `(n_observations, n_features)`
and distance computations should return a distance matrix: a symmetric ndarray of shape `(n_observations, n_observations)`
with zeros in the main diagonal.
Custom functions implementing clustering algorithms should return an object with a `labels_` attribute,
an ndarray of shape `(n_observations,)` that contains cluster indexes for each observation.

## Simple Usage
Define the distribution over clustering configurations in the options dict
```
options = {'model': {'choices': ['hac', 'kmeans']},
           'linkage': 'single',
           'metric': 'l2',
           'n_clusters': {'irange': [2, 10]},
           'transformation': {'choices': [None, 'svd'], 'weights': [1, 2]},
           'n_components': {'choices': [5, 10]}
           }
```
Fit a consensus clustering model \
`consensus_model = Consensus(X, options, precompute='distances').fit()`\

Sample and evaluate configurations \
`results = Evaluator(X, options, precompute='distances').random_search()` \

See `examples.py` for more applications.

## Dimensionality reduction
Implementations of an autoencoder with fully-connected layers and a convolutional autoencoder
are provided in `autoencoder.py`.
Both models estimate low dimensional codes by minimizing the reconstruction
loss of encoding and decoding through a lower dimensional bottleneck.

