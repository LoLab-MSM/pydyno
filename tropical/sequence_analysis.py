import pandas as pd
import numpy as np
import os
from collections import Iterable
from sklearn.metrics.pairwise import pairwise_distances
import editdistance
import tropical.lcs as lcs


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


class Sequences(object):
    def __init__(self, seqdata):
        # Checking seqdata
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

        self._sequences = data_seqs

        self._diss = None

    def unique_sequences(self):
        """
        Obtain the unique sequence in the dataframe
        Returns
        -------
        pd.DataFrame with the unique sequences

        """
        data_seqs = self._sequences.groupby(self._sequences.columns.tolist(),
                                            sort=False).size().rename('count').reset_index()
        data_seqs.set_index([list(range(len(data_seqs))), 'count'], inplace=True)
        return data_seqs

    def truncate_sequences(self, idx):
        """
        Truncates the sequences at the index passed
        Parameters
        ----------
        idx: int
            Index at which the sequences will be truncated

        Returns
        -------

        """
        data_seqs = self._sequences[self._sequences.columns.tolist()[:idx]]
        return data_seqs

    assign = lambda d, k: lambda f: d.setdefault(k, f)
    representativeness = {}

    def dissimilarity_matrix(self, metric='LCS', n_jobs=1):
        # Sort sequences
        sorted_seqs = self._sequences.sort_values(by=self._sequences.columns.tolist(), inplace=False)
        old_sorted_idxs = np.argsort(sorted_seqs.index)
        # Get unique sequences based on the sorted dataframe. This allows us to connect
        # the sorted dataframe with the unique sequences dataframes to relate them back
        # after obtaining the dissimilarity matrix
        unique_sequences = sorted_seqs.groupby(sorted_seqs.columns.tolist(),
                                               sort=False).size().rename('count').reset_index()
        unique_sequences.set_index([list(range(len(unique_sequences))), 'count'], inplace=True)

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
        self._diss = diss
        return diss

    @property
    def diss(self):
        if self._diss is None:
            print('dissimilarity_matrix function must be ran before'
                  'to obtain the values.')
        return self._diss

    @property
    def unique_states(self):
        unique_states = pd.unique(self._sequences[self._sequences.columns.tolist()].values.ravel())
        unique_states.sort()
        return unique_states

    @assign(representativeness, 'neighborhood')
    def neighborhood_density(self, proportion, sequences_idx=None):
        """
        Representativeness using neighborhood density method
        Parameters
        ----------
        proportion
        diss: dissimilarity matrix
        sequences_idx

        Returns
        -------

        """
        seq_len = self._sequences.shape[1]
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
            seqs = self._sequences.iloc[sequences_idx]
            seqs_diss = self._diss[sequences_idx][:, sequences_idx]
        else:
            seqs = self._sequences
            seqs_diss = self._diss
        seqs_neighbours = np.apply_along_axis(density, 1, seqs_diss)
        decreasing_seqs = seqs.iloc[seqs_neighbours.argsort()[::-1]]
        return decreasing_seqs.iloc[0].values

    @assign(representativeness, 'centrality')
    def centrality(self, sequences_idx=None):
        if sequences_idx is not None:
            seqs = self._sequences.iloc[sequences_idx]
            seqs_diss = self._diss[sequences_idx][:, sequences_idx]
        else:
            seqs = self._sequences
            seqs_diss = self._diss
        seqs_centrality_idx = seqs_diss.sum(axis=0).argsort()
        decreasing_seqs = seqs.iloc[seqs_centrality_idx]
        return decreasing_seqs.iloc[0].name, decreasing_seqs.iloc[0].values

    @assign(representativeness, 'frequency')
    def frequency(self, sequences_idx=None):
        decreasing_seqs = self.neighborhood_density(proportion=1, sequences_idx=sequences_idx)
        return decreasing_seqs

    def dispatch(self, k):  # , *args, **kwargs):
        try:
            method = self.representativeness[k].__get__(self, type(self))
        except KeyError:
            assert k in self.representativeness, "invalid operation: " + repr(k)
        return method  # (*args, **kwargs)

    def seq_representativeness(self, method='frequency', clus_labels=None, **kwargs):
        """

        Parameters
        ----------
        method
        clus_labels: vector-like
            Labels must be in the same order of seqdata
        kwargs

        Returns
        -------

        """
        rep_method = self.dispatch(method)
        if clus_labels is not None:
            clusters = set(clus_labels)
            clus_rep = {}
            for clus in clusters:
                clus_seqs = self._sequences.iloc[clus_labels == clus]
                clus_idxs = clus_seqs.index.get_level_values(0).values
                rep = rep_method(sequences_idx=clus_idxs, **kwargs)
                clus_rep[clus] = rep
            return clus_rep
        else:
            rep = rep_method(**kwargs)
            return rep


data = np.array([[11, 12, 13, 14, 15], [1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [6, 7, 8, 12, 11],
                 [11, 12, 13, 14, 15], [1, 2, 3, 4, 5]])
data_df = pd.DataFrame(data=data)
a = Sequences(data_df)
a.dissimilarity_matrix()
labels = np.array([0, 1, 2, 3, 0, 1])
a.seq_representativeness(clus_labels=labels)

