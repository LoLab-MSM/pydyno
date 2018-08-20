from __future__ import division
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import sklearn.cluster as cluster
from sklearn import metrics
from tropical.distinct_colors import distinct_colors
import tropical.lcs as lcs
from collections import Iterable
import editdistance
from tropical.kmedoids import kMedoids

try:
    import hdbscan
except ImportError:
    hdbscan = None

try:
    from pathos.multiprocessing import ProcessingPool as Pool
except ImportError:
    Pool = None

# Valid metrics from scikit-learn
_VALID_METRICS = ['euclidean', 'l2', 'l1', 'manhattan', 'cityblock',
                  'braycurtis', 'canberra', 'chebyshev', 'correlation',
                  'cosine', 'dice', 'hamming', 'jaccard', 'kulsinski',
                  'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto',
                  'russellrao', 'seuclidean', 'sokalmichener',
                  'sokalsneath', 'sqeuclidean', 'yule', "wminkowski"]


def lcs_dist_same_length(seq1, seq2):
    """

    Parameters
    ----------
    seq1 : array-like
        Sequence 1
    seq2 : array-like
        Sequence 2

    Returns
    -------

    """
    seq_len = len(seq1)
    d_1_2 = 2 * seq_len - 2 * lcs.lcs_std(seq1, seq2)[0]
    return d_1_2


def lcs_dist_diff_length(seq1, seq2):
    seq1_len = len(seq1)
    seq2_len = len(seq2)

    d_1_2 = seq1_len + seq2_len - 2 * lcs.lcs_std(seq1, seq2)[0]
    return d_1_2


def levenshtein(seq1, seq2):
    d_1_2 = editdistance.eval(seq1, seq2).__float__()
    return d_1_2


class ClusterSequences(object):
    """
    Class to cluster DynSign signatures

    Parameters
    ----------
    seqdata: str file, np.ndarray, or pandas dataframe
        file of pandas dataframe or ndarray where rows are DynSign signatures and columns are
         dominant states at specific  time points
    unique_sequences: bool, optional
        Drop repeated sequences
    truncate_seq: int
        Index of where to truncate a sequence
    """

    def __init__(self, seqdata, unique_sequences=False, truncate_seq=None):

        if isinstance(seqdata, str):
            if os.path.isfile(seqdata):
                data_seqs = pd.read_csv(seqdata, header=0, index_col=0)
                # convert column names into float numbers
                data_seqs.columns = [float(i) for i in data_seqs.columns.tolist()]
            else:
                raise TypeError('String is not a file')
        elif isinstance(seqdata, Iterable):
            data_seqs = pd.DataFrame(data=seqdata)
        elif isinstance(seqdata, pd.DataFrame):
            data_seqs = seqdata
        else:
            raise TypeError('data type not valid')

        # Truncate sequences at the index passed
        if isinstance(truncate_seq, int):
            data_seqs = data_seqs[data_seqs.columns.tolist()[:truncate_seq]]
        self.n_sequences = len(seqdata)

        # Get unique sequences
        if unique_sequences:
            self.sequences = self.get_unique_sequences(data_seqs)
            self.unique = True
        else:
            self.sequences = data_seqs
            self.unique = False

        # States in sequences
        unique_states = pd.unique(data_seqs[data_seqs.columns.tolist()].values.ravel())
        unique_states.sort()
        self.unique_states = unique_states

        self.diss = None
        self.labels = None
        self.cluster_method = ''

    def __repr__(self):
        return (
            '{} (Sequences:{}, Unique States:{})'.format(self.__class__.__name__, self.n_sequences, self.unique_states))

    @staticmethod
    def get_unique_sequences(data_seqs):
        data_seqs = data_seqs.groupby(data_seqs.columns.tolist(), sort=False).size().rename('count').reset_index()
        data_seqs.set_index([list(range(len(data_seqs))), 'count'], inplace=True)
        return data_seqs

    def diss_matrix(self, metric='LCS', n_jobs=1):
        """

        Parameters
        ----------
        metric : str, callable
            Metric to use to calculate the dissimilarity matrix
        n_jobs : int
            Number of processors to use

        Returns
        -------
        A Dissimilarity matrix
        """
        # TODO, there must be a better way to do this. Also, make sure that unique and complete sequences are the same
        if self.unique:
            unique_sequences = self.sequences.sort_values(by=self.sequences.columns.tolist(), inplace=False)
            old_sorted_idxs = np.argsort(unique_sequences.index)

            if metric in _VALID_METRICS:
                diss = pairwise_distances(unique_sequences.values, metric=metric, n_jobs=n_jobs)
            elif metric == 'LCS':
                diss = pairwise_distances(unique_sequences.values, metric=lcs_dist_same_length, n_jobs=n_jobs)
            elif metric == 'levenshtein':
                diss = pairwise_distances(unique_sequences.values, metric=levenshtein, n_jobs=n_jobs)
            elif callable(metric):
                diss = pairwise_distances(unique_sequences.values, metric=metric, n_jobs=n_jobs)
            else:
                raise ValueError('metric not supported')

            # This is to be able to match cluster idxs with the parameter idxs
            diss = diss[old_sorted_idxs]
            diss = diss.T
            diss = diss[old_sorted_idxs]
            self.diss = diss

        else:
            unique_sequences = self.sequences.sort_values(by=self.sequences.columns.tolist(), inplace=False)
            old_sorted_idxs = np.argsort(unique_sequences.index)
            unique_sequences = self.get_unique_sequences(unique_sequences)

            if metric in _VALID_METRICS:
                diss = pairwise_distances(unique_sequences.values, metric=metric, n_jobs=n_jobs)
            elif metric == 'LCS':
                diss = pairwise_distances(unique_sequences.values, metric=lcs_dist_same_length, n_jobs=n_jobs)
            elif metric == 'levenshtein':
                diss = pairwise_distances(unique_sequences.values, metric=levenshtein, n_jobs=n_jobs)
            elif callable(metric):
                diss = pairwise_distances(unique_sequences.values, metric=metric, n_jobs=n_jobs)
            else:
                raise ValueError('metric not supported')

            count_seqs = unique_sequences.index.get_level_values(1).values
            seq_idxs = unique_sequences.index.get_level_values(0).values
            repeat_idxs = np.repeat(seq_idxs, count_seqs)
            diss = diss[repeat_idxs]
            # This is to be able to math cluster idxs with the parameter idxs
            diss = diss[old_sorted_idxs]
            diss = diss.T
            diss = diss[repeat_idxs]
            diss = diss[old_sorted_idxs]
            self.diss = diss

    def hdbscan_clustering(self, min_cluster_size=50, min_samples=5,
                           alpha=1.0, cluster_selection_method='eom', **kwargs):
        """

        Parameters
        ----------
        min_cluster_size
        min_samples
        alpha
        cluster_selection_method
        kwargs

        Returns
        -------

        """
        if hdbscan is None:
            raise Exception('Please install the hdbscan package for this feature')
        if self.diss is None:
            raise Exception('Get the dissimilarity matrix first')
        hdb = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                              alpha=alpha, cluster_selection_method=cluster_selection_method,
                              metric='precomputed', **kwargs).fit(self.diss)
        self.labels = hdb.labels_

        self.cluster_method = 'hdbscan'
        return

    def Kmedoids(self, n_clusters):
        """

        Parameters
        ----------
        n_clusters : int
            Number of clusters

        Returns
        -------

        """
        if self.diss is None:
            raise Exception('Get the dissimilarity matrix first')
        kmedoids = kMedoids(self.diss, n_clusters)
        labels = np.empty(len(self.sequences), dtype=np.int32)
        for lb, seq_idx in kmedoids[1].items():
            labels[seq_idx] = lb
        self.cluster_method = 'kmedoids'
        self.labels = labels
        return

    def agglomerative_clustering(self, n_clusters, linkage='average', **kwargs):
        ac = cluster.AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed',
                                             linkage=linkage, **kwargs).fit(self.diss)
        self.labels = ac.labels_
        self.cluster_method = 'agglomerative'
        return

    def spectral_clustering(self, n_clusters, random_state=None, n_jobs=1, **kwargs):
        gamma = 1. / len(self.diss[0])
        kernel = np.exp(-self.diss * gamma)
        sc = cluster.SpectralClustering(n_clusters=n_clusters, random_state=random_state,
                                        affinity='precomputed', n_jobs=n_jobs, **kwargs).fit(kernel)
        self.labels = sc.labels_
        self.cluster_method = 'spectral'

    def silhouette_score(self):
        """

        Returns : Silhouette score to measure quality of the clustering
        -------

        """
        if self.labels is None:
            raise Exception('you must cluster the signatures first')
        if self.cluster_method == 'hdbscan':
            # Keep only clustered sequences
            clustered = np.where(self.labels != -1)[0]
            updated_labels = self.labels[clustered]
            updated_diss = self.diss[clustered][:, clustered]
            score = metrics.silhouette_score(updated_diss, updated_labels, metric='precomputed')
            return score
        else:
            score = metrics.silhouette_score(self.diss, self.labels, metric='precomputed')
            return score

    def silhouette_score_spectral_range(self, cluster_range, n_jobs=1, random_state=None, **kwargs):
        if isinstance(cluster_range, int):
            cluster_range = list(range(2, cluster_range + 1))  # +1 to cluster up to cluster_range
        elif hasattr(cluster_range, "__len__") and not isinstance(cluster_range, str):
            pass
        else:
            raise TypeError('Type not valid')

        gamma = 1. / len(self.diss[0])
        kernel = np.exp(-self.diss * gamma)
        cluster_silhouette = []
        for num_clusters in cluster_range:
            clusters = cluster.SpectralClustering(num_clusters, n_jobs=n_jobs, affinity='precomputed',
                                                  random_state=random_state, **kwargs).fit(kernel)
            score = metrics.silhouette_score(self.diss, clusters.labels_, metric='precomputed')
            cluster_silhouette.append(score)
        clusters_df = pd.DataFrame({'num_clusters': cluster_range, 'cluster_silhouette': cluster_silhouette})
        return clusters_df

    def __agg_cluster_score(self, num_clusters, linkage='average', **kwargs):
        clusters = cluster.AgglomerativeClustering(num_clusters, linkage=linkage,
                                                   affinity='precomputed', **kwargs).fit(self.diss)
        score = metrics.silhouette_score(self.diss, clusters.labels_, metric='precomputed')
        return score

    def silhouette_score_agglomerative_range(self, cluster_range, linkage='average', n_jobs=1, **kwargs):
        """

        Parameters
        ----------
        cluster_range : list-like or int
            Range of the number of clusterings to obtain the silhouette score
        linkage : str
            Type of agglomerative linkage
        kwargs : key arguments to pass to the aggomerative clustering function

        Returns
        -------

        """

        if isinstance(cluster_range, int):
            cluster_range = list(range(2, cluster_range + 1))  # +1 to cluster up to cluster_range
        elif hasattr(cluster_range, "__len__") and not isinstance(cluster_range, str):
            pass
        else:
            raise TypeError('Type not valid')
        if n_jobs == 1:
            cluster_silhouette = []
            for num_clusters in cluster_range:
                score = self.__agg_cluster_score(num_clusters, linkage=linkage, **kwargs)
                cluster_silhouette.append(score)
            clusters_df = pd.DataFrame({'num_clusters': cluster_range, 'cluster_silhouette': cluster_silhouette})
            return clusters_df
        else:
            if Pool is None:
                raise Exception('Please install the pathos package for this feature')
            p = Pool(n_jobs)
            res = p.amap(lambda x: self.__agg_cluster_score(x, linkage, **kwargs), cluster_range)
            scores = res.get()
            clusters_df = pd.DataFrame({'num_clusters': cluster_range, 'cluster_silhouette': scores})
            return clusters_df

    def calinski_harabaz_score(self):
        if self.labels is None:
            raise Exception('you must cluster the signatures first')
        score = metrics.calinski_harabaz_score(self.sequences, self.labels)
        return score

    #TODO: Develop gap statistics score for sequences. Maybe get elbow plot as well

    assign = lambda d, k: lambda f: d.setdefault(k, f)
    representativeness = {}

    @assign(representativeness, 'neighborhood')
    def neighborhood_density(self, proportion, sequences_idx=None):
        seq_len = self.sequences.shape[1]
        ci = 1  # ci is the indel cost
        s = 2  # s is the substitution cost
        # this is the maximal distance between two sequences using the optimal matching metric
        # with indel cost ci=1 and substitution cost s=2. Gabardinho et al (2011) communications in computer
        # and information science
        theo_max_dist = seq_len * min([2 * ci, s])
        neighbourhood_radius = theo_max_dist * proportion

        def density(seq_dists):
            seq_density = len(seq_dists[seq_dists < neighbourhood_radius])
            return seq_density

        if sequences_idx is not None:
            seqs = self.sequences.iloc[sequences_idx]
            seqs_diss = self.diss[sequences_idx][:, sequences_idx]
        else:
            seqs = self.sequences
            seqs_diss = self.diss
        seqs_neighbours = np.apply_along_axis(density, 1, seqs_diss)
        decreasing_seqs = seqs.iloc[seqs_neighbours.argsort()[::-1]]
        return decreasing_seqs.iloc[0].values

    @assign(representativeness, 'centrality')
    def centrality(self, sequences_idx=None):
        if sequences_idx is not None:
            seqs = self.sequences.iloc[sequences_idx]
            seqs_diss = self.diss[sequences_idx][:, sequences_idx]
        else:
            seqs = self.sequences
            seqs_diss = self.diss
        seqs_centrality_idx = seqs_diss.sum(axis=0).argsort()
        decreasing_seqs = seqs.iloc[seqs_centrality_idx]
        return decreasing_seqs.iloc[0].name, decreasing_seqs.iloc[0].values

    @assign(representativeness, 'frequency')
    def frequency(self, sequences_idx=None):
        decreasing_seqs = self.neighborhood_density(proportion=1, sequences_idx=sequences_idx)
        return decreasing_seqs

    # def transition_rate_matrix(self, time_varying=False, lag=1):
    #     # this code comes from seqtrate from the TraMineR package in r
    #     nbetat = len(self.unique_states)
    #     sdur = self.sequences.shape[1]
    #     alltransitions = np.arange(0, sdur - lag)
    #     numtransition = len(alltransitions)
    #     row_index = pd.MultiIndex.from_product([alltransitions, self.unique_states],
    #                                            names=['time_idx', 'from_state'])  # , names=row_names)
    #     col_index = pd.MultiIndex.from_product([self.unique_states], names=['to_state'])  # , names=column_names)
    #     if time_varying:
    #         array_zeros = np.zeros(shape=(nbetat * numtransition, nbetat))
    #         tmat = pd.DataFrame(array_zeros, index=row_index, columns=col_index)
    #         for sl in alltransitions:
    #             for x in self.unique_states:
    #                 colxcond = self.sequences[[sl]] == x
    #                 PA = colxcond.sum().values[0]
    #                 if PA == 0:
    #                     tmat.loc[sl, x] = 0
    #                 else:
    #                     for y in self.unique_states:
    #                         PAB_p = np.logical_and(colxcond, self.sequences[[sl + lag]] == y)
    #                         PAB = PAB_p.sum().values[0]
    #                         tmat.loc[sl, x][[y]] = PAB / PA
    #     else:
    #         tmat = pd.DataFrame(index=self.unique_states, columns=self.unique_states)
    #         for x in self.unique_states:
    #             # PA = 0
    #             colxcond = self.sequences[alltransitions] == x
    #             if numtransition > 1:
    #                 PA = colxcond.sum(axis=1).sum()
    #             else:
    #                 PA = colxcond.sum()
    #             if PA == 0:
    #                 tmat.loc[x] = 0
    #             else:
    #                 for y in self.unique_states:
    #                     if numtransition > 1:
    #                         PAB_p = np.logical_and(colxcond, self.sequences[alltransitions + lag] == y)
    #                         PAB = PAB_p.sum(axis=1).sum()
    #                     else:
    #                         PAB_p = np.logical_and(colxcond, self.sequences[alltransitions + lag] == y)
    #                         PAB = PAB_p.sum()
    #                     tmat.loc[x][[y]] = PAB / PA
    #
    #     return tmat

    # def seqlogp(self, prob='trate', time_varying=True, begin='freq'):
    #     sl = self.sequences.shape[1]  # all sequences have the same length for our analysis
    #     maxage = sl
    #     nbtrans = maxage - 1
    #     agedtr = np.zeros(maxage)

    def dispatch(self, k):  # , *args, **kwargs):
        try:
            method = self.representativeness[k].__get__(self, type(self))
        except KeyError:
            assert k in self.representativeness, "invalid operation: " + repr(k)
        return method  # (*args, **kwargs)

    def cluster_percentage_color(self, representative_method='centrality', **kwargs):
        if self.labels is None:
            raise Exception('you must cluster the signatures first')

        rep_method = self.dispatch(representative_method)
        clusters = set(self.labels)
        colors = distinct_colors(len(clusters))
        cluster_inf = {}
        for clus in clusters:
            clus_seqs = self.sequences.iloc[self.labels == clus]
            clus_idxs = clus_seqs.index.get_level_values(0).values
            rep = rep_method(sequences_idx=clus_idxs, **kwargs)
            n_seqs = clus_seqs.shape[0]
            # This is to sum over the index of sequences that have the sequence repetitions
            if self.unique:
                total_seqs = 0
                for seq in clus_seqs.index.values:
                    total_seqs += seq[1]
            else:
                total_seqs = n_seqs

            cluster_percentage = total_seqs / self.n_sequences
            cluster_inf[clus] = (cluster_percentage, colors[clus], rep)

        return cluster_inf
