from sklearn import preprocessing
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.metrics import pairwise_distances
from sklearn.cluster import (AgglomerativeClustering,
                             DBSCAN,
                             MiniBatchKMeans,
                             KMeans)

from autoencoders import Autoencoder, ConvolutionalAutoencoder


def hac(params, X):
    '''
    Fit and return an AgglomerativeClustering model.

    Parameters:
        params : dict
            The parameters of the model.
        X : ndarray of shape (n_observations, n_observations)
            or (n_observation, n_features)
            Distance or features matrix of observations to be clustered.
    '''
    if 'metric' in params:
        del params['metric']
        params['affinity'] = 'precomputed'
    return AgglomerativeClustering(**params).fit(X)

def hacsingle(params, X):
    '''
    Fit and return an AgglomerativeClustering model with single linkage.

    Parameters:
        params : dict
            The parameters of the model.
        X : ndarray of shape (n_observations, n_observations)
            or (n_observation, n_features)
            Distance or features matrix of observations to be clustered.
    '''
    params['linkage'] = 'single'
    return hac(params, X)

def hacaverage(params, X):
    '''
    Fit and return an AgglomerativeClustering model with average linkage.

    Parameters:
        params : dict
            The parameters of the model.
        X : ndarray of shape (n_observations, n_observations)
            or (n_observation, n_features)
            Distance or features matrix of observations to be clustered.
    '''
    params['linkage'] = 'average'
    return hac(params, X)

def mbkmeans(params, X):
    '''
    Fit and return a MiniBatchKMeans model.

    Parameters:
        params : dict
            The parameters of the model.
        X : ndarray of shape (n_observations, n_features)
            The data to be clustered.
    '''
    return MiniBatchKMeans(**params).fit(X)

def kmeans(params, X):
    '''
    Fit and return a KMeans model.

    Parameters:
        params : dict
            The parameters of the model.
        X : ndarray of shape (n_observations, n_features)
            The data to be clustered.
    '''
    return KMeans(**params).fit(X)

def dbscan(params, X):
    '''
    Fit and return a DBSCAN model.

    Parameters:
        params : dict
            The parameters of the model.
        X : ndarray of shape (n_observations, n_observations)
            or (n_observation, n_features)
            Distance or features matrix of observations to be clustered.
    '''
    if 'metric' in params:
        params['metric'] = 'precomputed'
    return DBSCAN(**params).fit(X)

def svd(params, X):
    '''
    Fit a TruncatedSVD model on l2-normalized data and return the
    transformed data.

    Parameters:
        params : dict
            The parameters of the model.
        X : ndarray of shape (n_observations, n_features)
            The data to fit the model and be transformed.
    '''
    X = preprocessing.normalize(X)
    return TruncatedSVD(**params).fit_transform(X)

def autoencoder(params, X):
    '''
    Fit an Autoencoder model and return transformed and l2-normalized data.

    Parameters:
        params : dict
            The parameters of the model.
        X : ndarray of shape (n_observations, n_features)
            The data to fit the model and be transformed.
    '''
    ae = Autoencoder(**params)
    ae.fit(X)
    X = ae.encode(X)
    return preprocessing.normalize(X)

def convautoencoder(params, X):
    '''
    Fit a ConvoluationalAutoencoder model and return transformed
    and l2-normalized data.

    Parameters:
        params : dict
            The parameters of the model.
        X : ndarray of shape (n_observations, width, height, channels)
            The data to fit the model and be transformed.
    '''
    cae = ConvolutionalAutoencoder(**params)
    cae.fit(X)
    X = cae.encode(X)
    return preprocessing.normalize(X)

def rbf(params, X):
    '''
    Compute distances with an rbf metric.

    Parameters:
        params : dict
            Parameters for computing the rbf distances.
        X : ndarray of shape (n_observations, n_features)
            Feature matrix to be transformed into distance matrix.

    Returns:
        X_distances: ndarray of shape (n_observations, n_observations)
            Distances between observations.
    '''
    return 1. - pairwise_kernels(X, metric='rbf', **params)


class Executor:
    '''
    Compute clustering models, feature transformations and distances.

    Parameters:
        definitions : dict
            Dict mapping function names to function definitions.
            Extends and/or overwrites the default definitions.
    '''

    definitions = {'hac': hac,
                   'hacsingle': hacsingle,
                   'hacaverage': hacaverage,
                   'kmeans': kmeans,
                   'mbkmeans': mbkmeans,
                   'dbscan': dbscan,
                   'rbf': rbf,
                   'svd': svd,
                   'autoencoder': autoencoder,
                   'convautoencoder': convautoencoder}

    def __init__(self, definitions=None):
        if definitions is not None:
            self.definitions = {**self.definitions, **definitions}

    def fit_model(self, config, X):
        '''
        Fit and return a clustering model.

        Parameters:
            config : dict
                Configuration of the clustering.
            X : ndarray of shape (n_observations, n_features)
                or (n_observations, n_observations)
                The data to be clustered.
        '''
        model = self.definitions[config['model']]
        return model(config['model_parameters'], X)

    def transform(self, config, X):
        '''
        Compute and return transformated features.

        Parameters:
            config : dict
                Configuration of the clustering.
            X : ndarray of shape (n_observations, n_features)
                The data to be transformed.
        '''
        func = self.definitions[config['transformation']]
        return func(config['transformation_parameters'], X)

    def compute_distances(self, config, X):
        '''
        Compute and return distances between observations.

        Parameters:
            config : dict
                Configuration of the clustering.
            X : ndarray of shape (n_observations, n_features)
                The data to compute distances from.
        '''
        metric = config['metric']
        func = self.definitions.get(metric, None)
        if func is not None:
            return func(config['metric_parameters'], X)
        return pairwise_distances(X, metric=metric)
