import numpy as np

from sklearn.metrics import (silhouette_score,
                             calinski_harabasz_score,
                             davies_bouldin_score,
                             adjusted_rand_score,
                             adjusted_mutual_info_score,
                             normalized_mutual_info_score,
                             fowlkes_mallows_score)

from scipy.optimize import linear_sum_assignment

from cluster_generation import ClusterGenerator


def ground_truth_metrics(y_true, y_pred):
    '''Compute cluster evaluation metrics that require ground truth labels.'''
    return {'accuracy': Scorer.accuracy(y_true, y_pred),
            'adjusted_rand_index': Scorer.ari(y_true, y_pred),
            'adjusted_mutual_information': Scorer.ami(y_true, y_pred),
            'normalized_mutual_information': Scorer.nmi(y_true, y_pred),
            'fscore': Scorer.fscore(y_true, y_pred)}

def consensus_metrics(consensus, y_true=None):
    '''Compute metrics for a consensus model.'''
    y_pred = consensus.model.labels_
    metrics = {
        'num_clusters': len(set(y_pred)),
        'informativeness': Scorer.informativeness(consensus.consensus_matrix)
    }
    if y_true is not None:
        metrics = {**metrics, **ground_truth_metrics(y_true, y_pred)}
    return metrics

def _normalize_similarity_matrix(K):
    N = np.diag(1. / np.sqrt(np.diagonal(K)))
    return N @ K @ N


class Scorer:
    '''Computate clustering evaluation metrics.'''

    @staticmethod
    def score(clustering, criterion, y_true=None):
        '''Select a scoring function based on criterion and return the result.'''
        scoring_func = getattr(Scorer, criterion)

        if criterion == 'silhouette':
            metric = clustering['config'].get('metric', None) or 'l2'
            return scoring_func(clustering['data'],
                                clustering['model'].labels_,
                                metric)

        if criterion in ('chscore', 'dbindex'):
            return scoring_func(clustering['data'], clustering['model'].labels_)

        if criterion in ('accuracy', 'ari', 'ami', 'nmi', 'fscore'):
            indexes = clustering['config'].get('indexes', None)
            if indexes is not None:
                y_true = y_true[indexes]
            return scoring_func(y_true, clustering['model'].labels_)

    @staticmethod
    def accuracy(y_true, y_pred):
        '''Compute clustering accuracy.'''
        K = max(y_pred.max(), y_true.max()) + 1
        W = np.zeros((K, K), dtype='int64')
        for i in range(len(y_pred)):
            W[y_pred[i], y_true[i]] += 1
        rows, cols = linear_sum_assignment(W.max() - W)
        return np.sum(W[rows, cols]) / len(y_pred)

    @staticmethod
    def ari(y_true, y_pred):
        '''Compute adjusted rand index.'''
        return adjusted_rand_score(y_true, y_pred)

    @staticmethod
    def ami(y_true, y_pred):
        '''Compute adjusted mutual information.'''
        return adjusted_mutual_info_score(y_true, y_pred)

    @staticmethod
    def nmi(y_true, y_pred):
        '''Compute normalized mutual information.'''
        return normalized_mutual_info_score(y_true, y_pred)

    @staticmethod
    def fscore(y_true, y_pred):
        '''Compute fscore.'''
        return fowlkes_mallows_score(y_true, y_pred)

    @staticmethod
    def silhouette(X, y_pred, metric):
        '''Compute silhouette score.'''
        return silhouette_score(X, y_pred, metric=metric)

    @staticmethod
    def chscore(X, y_pred):
        '''Compute calinski harabasz score.'''
        return calinski_harabasz_score(X, y_pred)

    @staticmethod
    def dbscore(X, y_pred):
        '''Compute davies boulding score.'''
        return -davies_bouldin_score(X, y_pred)

    @staticmethod
    def informativeness(K, normalize=False, measure='cosine'):
        '''Compute informativeness of a similarity matrix.'''
        if normalize:
            K = _normalize_similarity_matrix(K)
        n = len(K)
        k = np.mean(K)
        Kf = np.linalg.norm(K, 'fro')
        if measure == 'cosine':
            return 1. - (n / Kf) * np.sqrt((n*k**2 - 2*k + 1) / (n-1))
        if measure == 'euclidean':
            return np.sqrt(Kf**2 / n**2 - ((n*k**2 - 2*k + 1) / (n-1)))


class Evaluator:
    '''Evaluate clusterings from a ClusterGenerator.'''

    def __init__(self,
                 X,
                 options,
                 parameters=None,
                 definitions=None,
                 features=None,
                 distances=None,
                 precompute=None,
                 labels=None,
                 verbose=True):
        self.generator = ClusterGenerator(X,
                                          options=options,
                                          parameters=parameters,
                                          definitions=definitions,
                                          features=features,
                                          distances=distances,
                                          precompute=precompute)
        self.labels = labels
        self.verbose = verbose

    def evaluate_configs(self, configs, criteria=['silhouette']):
        '''
        Evaluate clusterings from configs.
        
        Parameters:
            configs : array-like of dict
                The configs to be evaluated.
            criteria: array-like of str
                The evaluation criteria to be computed.
        
        Returns:
            results : list of dict
                The configs and score for each criterion.
        '''
        results = []
        for i, config in enumerate(configs):

            if self.verbose:
                print('\rConfig {}/{}'.format(i+1, len(configs)),
                      end='\n' if i == len(configs)-1 else '',
                      flush=True)

            clustering = self.generator.cluster(config)
            scores = {criterion: Scorer.score(clustering, criterion, self.labels)
                      for criterion in criteria}
            results.append({'config': config, 'scores': scores})

        return results

    def random_search(self, n_samples=100, criterion='silhouette'):
        '''
        Evaluate clusterings from config samples.
        
        Parameters:
            n_samples : int
                The number of samples to be evaluated.
            criterion: str
                The evaluation criterion to be computed.
                
        Returns:
            results : dict
                A dict containing the best model and the sampled configs with
                their score.
        '''
        scored_configs = []
        best_score = -np.inf
        best_model = None

        for i in range(n_samples):

            if self.verbose:
                print('\rSample {}/{}'.format(i+1, n_samples),
                      end='\n' if i == n_samples-1 else '',
                      flush=True)

            clustering = self.generator.sample()
            score = Scorer.score(clustering, criterion, self.labels)
            scored_configs.append((clustering['config'], score))

            if score > best_score:
                best_score = score
                best_model = clustering['model']

        scored_configs.sort(key=lambda x: x[1], reverse=True)

        return {'best_model': best_model, 'scored_configs': scored_configs}
