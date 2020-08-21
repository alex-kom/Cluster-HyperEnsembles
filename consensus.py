import numpy as np

from cluster_generation import ClusterGenerator
from backend import hac
from evaluation import Scorer


class Consensus:
    '''
    Fit a cluster ensemble model by aggregating cluster assignments of
    randomized clusterings.

    The randomized clusterings are produced by a ClusterGenerator object.
    The ensemble is fitted by computing a consensus similarity matrix from
    cluster assignments. The final clustering is performed by hierarchical
    agglomerative clustering on the consensus simililarity matrix.
    The silhouette score is used to estimate the number of clusters.

    Arguments:
        X : ndarray of shape (n_observations, n_features).
            The data to be clustered.
        options : dict
            Dict specifying the parameter values for the clustering
            algorithms, the feature transformations, the distance metrics,
            and the data subsampling rate. Values of type dict specify
            a categorical distribution over possible values.
        parameters : dict
            Dict specifying the required parameters for the clustering
            algorithms, feature transformations and distance metrics.
            Extends and/or overwrites the default parameters.
        definitions : dict
            Dict mapping function names of clustering algorithms,
            feature transformations, and distance metrics to function
            objects. Extends and/or overwrites the default definitions.
        features : dict
            Dict mapping feature config strings to feature matrices.
            Feature matrices are ndarrays of shape (n_observations, n_features).
        distances : dict
            Dict mapping distance config strings to distance matrices.
            Distance matrices are ndarrays of shape
            (n_observations, n_observations).
        precompute : {'features', 'distances', None}
            If 'features', all the feature transformations
            are precomputed and stored in the features dictionary.
            If 'distances', all feature transformations and all distances
            are precomputed and stored in the distances dictionary.
        n_samples : int
            The number of clusterings to be sampled for the consensus.
        mode : {'similarity', 'clustering'}
            If 'similarity', only the consensus similarity matrix will
            be estimated. If 'clustering', the consensus similarity
            matrix will be used to fit a hierarchical agglomerative clustering
            model and the silhouette score will be used to choose the number
            of clusters.
        linkage : {'average', 'single'}
            The linkage used in the hierarchical agglomerative clustering of the
            consensus matrix.
        verbose : Boolean
            If True, a message of progress will be printed during estimation.
    '''

    def __init__(self,
                 X,
                 options,
                 parameters=None,
                 definitions=None,
                 features=None,
                 distances=None,
                 precompute=None,
                 n_samples=100,
                 mode='cluster',
                 linkage='average',
                 verbose=True):
        self.generator = ClusterGenerator(X,
                                          options=options,
                                          parameters=parameters,
                                          definitions=definitions,
                                          features=features,
                                          distances=distances,
                                          precompute=precompute)
        self.n_samples = n_samples
        self.mode = mode
        self.linkage = linkage
        self.verbose = verbose
        self.n_clusters = self._get_nclusters()
        data_size = len(X)
        self.indexes = np.arange(data_size)
        self.consensus_matrix = np.zeros((data_size, data_size))
        self.counts = np.zeros((data_size, data_size))

    def _update_consensus(self, labels, indexes):
        if indexes is None:
            indexes = self.indexes
            self.counts += 1
        else:
            rows, cols = _get_product_indexes(indexes, indexes)
            self.counts[rows, cols] += 1

        for label in set(labels):
            if label != -1:
                l_indexes = indexes[np.where(labels == label)[0]]
                rows, cols = _get_product_indexes(l_indexes, l_indexes)
                self.consensus_matrix[rows, cols] += 1

    def _normalize_consensus(self):
        self.consensus_matrix /= np.maximum(self.counts, 1)

    def _fit_consensus_clusters(self):
        D = 1. - self.consensus_matrix
        np.fill_diagonal(D, 0)
        params = {'linkage': self.linkage, 'affinity': 'precomputed'}
        self.score = -1
        self.model = None

        for idx, n in enumerate(self.n_clusters, 1):

            if self.verbose:
                print('\rEstimating number of clusters {}/{}'.format(
                        idx, len(self.n_clusters)),
                      end='\n' if idx == len(self.n_clusters) else '',
                      flush=True)

            params['n_clusters'] = n
            model = hac(params, D)
            score = Scorer.silhouette(D, model.labels_, 'precomputed')

            if score > self.score:
                self.model = model
                self.score = score

    def _get_nclusters(self):
        n_clusters = self.generator.configs.options.get('n_clusters', None)
        if isinstance(n_clusters, dict):
            if 'irange' in n_clusters:
                return range(*n_clusters['irange'])
            if 'choices' in n_clusters:
                return n_clusters['choices']
        if isinstance(n_clusters, int):
            return [n_clusters]

    def fit(self):
        '''
        Fit a consensus clustering model.

        Return the consensus similarity matrix if mode == 'similarity' or an
        AgglomerativeClustering object fitted on the consensus similarity
        matrix if mode == 'cluster'.
        '''
        for i in range(self.n_samples):

            if self.verbose:
                print('\rSample {}/{}'.format(i+1, self.n_samples),
                      end='\n' if i == self.n_samples-1 else '',
                      flush=True)

            clustering_instance = self.generator.sample()
            model = clustering_instance['model']
            indexes = clustering_instance['config']['sample_indexes']
            self._update_consensus(model.labels_, indexes)

        self._normalize_consensus()

        if self.mode == 'cluster':
            self._fit_consensus_clusters()
            return self.model

        if self.mode == 'similarity':
            return self.consensus_matrix


def _get_product_indexes(x, y):
    return np.tile(x, len(y)), np.repeat(y, len(x))
