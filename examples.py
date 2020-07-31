import numpy as np
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from itertools import cycle, islice

from sklearn import datasets, preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from consensus import Consensus
from evaluation import ground_truth_metrics, consensus_metrics
from autoencoders import Autoencoder, ConvolutionalAutoencoder


def print_metrics(metrics):
    '''Print the evaluation metrics.'''
    for k, v in metrics.items():
        print('{}: {}'.format(k, v))

def mnist_cae(params, X):
    '''
    Fit a ConvolutionalAutoencoder and return encoded and l2-normalized data.
    '''
    X = X / 255.
    X = X.reshape(-1, 28, 28, 1)
    cae = ConvolutionalAutoencoder(
            kernel_shapes=[(5, 5, 64), (5, 5, 64), (3, 3, 64)],
            **params
            )
    cae.fit(X)
    X = cae.encode(X)
    X = preprocessing.normalize(X)
    return X

def tfidf_ae(params, X):
    '''
    Fit an Autoencoder on tf-idf transformed features and return encoded
    and l2-normalized data.
    '''
    X = TfidfTransformer(sublinear_tf=True).fit_transform(X).toarray()
    ae = Autoencoder(**params)
    ae.fit(X)
    X = ae.encode(X)
    X = preprocessing.normalize(X)
    return X


def cluster_artificial():
    '''
    Generate and cluster artificial data, print evaluation metrics and plot
    the clustered data.

    Based on the clustering comparison from
    https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html
    '''
    options = {'model': {'choices': ['hacsingle', 'hacaverage',
                                     'kmeans', 'dbscan']},
               'metric': 'rbf',
               'gamma': {'choices': [0.001, 0.01, 0.1, 1.]},
               'n_clusters': {'min_val': 2, 'max_val': 20},
               'eps': {'choices': [0.1, 0.5, 1.]},
               'min_samples': {'min_val': 5, 'max_val': 20},
               'data_subsampling_rate': 0.2}

    n_samples = 1500
    data = [datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05),
            datasets.make_moons(n_samples=n_samples, noise=.05),
            datasets.make_blobs(n_samples=n_samples, random_state=8),
            (np.random.rand(n_samples, 2), None),
            datasets.make_blobs(n_samples=n_samples,
                                cluster_std=[1.0, 2.5, 0.5],
                                random_state=170)]
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=170)
    data.append((np.dot(X, [[0.6, -0.6], [-0.4, 0.8]]), y))

    plt.figure(figsize=(21, 12.5))
    plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                        hspace=.01)

    for idx, (X, y) in enumerate(data, 1):
        print('Dataset {}/{}'.format(idx, len(data)))

        consensus = Consensus(X, options, precompute='distances', linkage='single')
        consensus.fit()
        print_metrics(consensus_metrics(consensus, y))
        y_pred = consensus.model.labels_

        X = preprocessing.StandardScaler().fit_transform(X)
        plt.subplot(3, 2, idx)
        colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(y_pred) + 1))))
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])
        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        plt.xticks(())
        plt.yticks(())

    plt.show()

def cluster_mnist():
    '''
    Download and cluster the test subset of mnist with a cluster ensemble.
    Print evaluation metrics for the ensemble and for a kmeans baseline.
    '''
    r = requests.get('https://s3.amazonaws.com/img-datasets/mnist.npz')
    data = np.load(BytesIO(r.content))
    X = data['x_test']
    y = data['y_test']

    definitions = {'mnist_cae': mnist_cae}
    parameters = {'mnist_cae': ['h_dim']}
    options = {'model': {'choices': ['hacaverage', 'mbkmeans']},
               'transformation': 'mnist_cae',
               'metric': 'rbf',
               'gamma': {'choices': [0.001, 0.01, 0.1, 1.]},
               'n_clusters': {'min_val': 2, 'max_val': 50},
               'h_dim': {'choices': [16, 24, 32]},
               'data_subsampling_rate': 0.2
               }

    consensus = Consensus(X,
                          options,
                          parameters=parameters,
                          definitions=definitions,
                          precompute='distances')
    consensus.fit()
    print_metrics(consensus_metrics(consensus, y))

    # kmeans baseline with 10 clusters
    baseline_config = {'model': 'mbkmeans',
                       'model_parameters': {'n_clusters': 10},
                       'transformation': 'mnist_cae',
                       'transformation_parameters': {'h_dim': 24},
                       'sample_indexes': None}
    baseline_model = consensus.generator.cluster(baseline_config)['model']
    print_metrics(ground_truth_metrics(y, baseline_model.labels_))

def cluster_newsgroups():
    '''
    Download and cluster the test subset of 20 newsgroups with a cluster
    ensemble. Print evaluation metrics for the ensemble and for a kmeans baseline.
    '''
    data = datasets.fetch_20newsgroups(subset='test',
                                       remove=('headers', 'footers', 'quotes'))
    X = CountVectorizer(max_features=5000).fit_transform(data['data']).toarray()
    y = data['target']

    options = {'model': 'mbkmeans',
               'transformation': 'tfidf_ae',
               'n_clusters': {'min_val': 2, 'max_val': 50},
               'layer_dims': {'choices':[(512, 512, 512, 32),
                                         (512, 512, 512, 48),
                                         (512, 512, 512, 64)]},
               'data_subsampling_rate': 0.2}
    parameters = {'tfidf_ae': ['layer_dims']}
    definitions = {'tfidf_ae': tfidf_ae}

    consensus = Consensus(X, options,
                          parameters=parameters,
                          definitions=definitions,
                          precompute='features')
    consensus.fit()
    print_metrics(consensus_metrics(consensus, y))

    # kmeans baseline with 20 clusters
    baseline_config = {'model': 'mbkmeans',
                       'model_parameters': {'n_clusters': 20},
                       'transformation': 'tfidf_ae',
                       'transformation_parameters':
                           {'layer_dims': (512, 512, 512, 32)},
                       'sample_indexes': None}
    baseline_model = consensus.generator.cluster(baseline_config)['model']
    print_metrics(ground_truth_metrics(y, baseline_model.labels_))


if __name__ == '__main__':
    cluster_artificial()
    cluster_mnist()
    cluster_newsgroups()
